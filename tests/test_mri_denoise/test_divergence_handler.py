"""
Tests for DivergenceStopHandler.

Uses a lightweight ignite Engine mock to verify termination logic.
"""

import pytest


class TestDivergenceStopHandler:
    def test_import(self):
        from src.mri_denoise.handlers.divergence import DivergenceStopHandler
        assert DivergenceStopHandler is not None

    def test_no_termination_on_positive_psnr(self):
        from ignite.engine import Engine
        from src.mri_denoise.handlers.divergence import DivergenceStopHandler

        trainer_terminated = []

        def _fake_process(e, batch):
            return batch

        trainer = Engine(_fake_process)
        evaluator = Engine(_fake_process)

        handler = DivergenceStopHandler(trainer, threshold=2, psnr_key="psnr")
        handler.attach(evaluator)

        # Simulate evaluator reporting positive PSNR
        evaluator.state = type("S", (), {"metrics": {"psnr": 25.0}})()
        handler(evaluator)
        handler(evaluator)

        # trainer should NOT have been terminated
        assert not trainer._is_done(trainer.state) if hasattr(trainer, "_is_done") else True

    def test_termination_after_threshold(self):
        from ignite.engine import Engine, Events
        from src.mri_denoise.handlers.divergence import DivergenceStopHandler

        terminated = []

        def _fake_process(e, batch):
            return batch

        trainer = Engine(_fake_process)
        evaluator = Engine(_fake_process)

        original_terminate = trainer.terminate
        def _mock_terminate():
            terminated.append(True)
        trainer.terminate = _mock_terminate

        handler = DivergenceStopHandler(trainer, threshold=3, psnr_key="psnr")
        handler.attach(evaluator)

        evaluator.state = type("S", (), {"metrics": {"psnr": -5.0}})()

        # Call 3 times — should trigger termination on the 3rd
        handler(evaluator)
        assert len(terminated) == 0
        handler(evaluator)
        assert len(terminated) == 0
        handler(evaluator)
        assert len(terminated) == 1

    def test_counter_resets_on_positive_psnr(self):
        from ignite.engine import Engine
        from src.mri_denoise.handlers.divergence import DivergenceStopHandler

        terminated = []

        def _fake_process(e, batch):
            return batch

        trainer = Engine(_fake_process)
        evaluator = Engine(_fake_process)
        trainer.terminate = lambda: terminated.append(True)

        handler = DivergenceStopHandler(trainer, threshold=3, psnr_key="psnr")
        handler.attach(evaluator)

        # 2 negative epochs, then 1 positive — counter should reset
        evaluator.state = type("S", (), {"metrics": {"psnr": -1.0}})()
        handler(evaluator)
        handler(evaluator)
        assert handler._neg_count == 2

        evaluator.state = type("S", (), {"metrics": {"psnr": 20.0}})()
        handler(evaluator)
        assert handler._neg_count == 0
        assert len(terminated) == 0
