import pytest
import flaky

# Number of objects (Test or Function) 8
# Number of functions 2
# Number of tests 6
# Number of Flaky Tests 5
# Number of Non Flaky Tests 1

@pytest.fixture(scope="class")
def fixtureSample():
    return

class TestNoFlaky():

    def test_nonFlaky(self):
        pass
    
    @flaky(max_runs=5, min_passes=3)
    def test_flakyParamFunc(self):
        pass

    @flaky
    def test_flakySimpleFunc(self):
        pass

    @flaky
    async def test_flakySimpleAsyncFunc(self):
        pass

@flaky
class TestFlaky():

    def test_flakyFuncBecauseClass(self):
        pass

    @pytest.mark.parametrize("angle", [0.0, 0.3, 0.5, 0.7, -0.2, 2.4])
    def test_parametrized(self):
        pass


class Function():

    def simpleFunction(self):
        pass

    async def simpleAsyncFunction(self):
        pass