from hysteron_tgraphs import candidategraphs

class TestScaffoldGeneration:
    def test_scaffold_count_is_correct(self):
        num_hysts = 2

        scaffold_count = len([scaffold for scaffold in candidategraphs.make_all_scaffolds(num_hysts)])

        assert scaffold_count == 2

    def test_analytical_scaffold_count_is_correct(self):
        num_hysts = 2

        analytical_scaffold_count = candidategraphs.count_scaffolds(num_hysts)

        assert analytical_scaffold_count == 2