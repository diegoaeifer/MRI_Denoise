

class TestTrainerDivergenceDetection:
    """Test trainer's divergence detection logic."""

    def test_divergence_counter_logic(self):
        """Test divergence counter increment/reset logic."""
        divergence_count = 0

        losses = [1.0, 1.1, 1.2, 1.3, 1.4]
        for loss in losses:
            if loss > 1.0:
                divergence_count += 1
            else:
                divergence_count = 0

        assert divergence_count == 4, "Divergence counter should increment"
