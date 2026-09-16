
@pytest.fixture()
def monocle_test_case(test_case: Union[TestCase|dict], request:pytest.FixtureRequest):
        validator = MonocleValidator()
        if isinstance(test_case, dict):
            test_case = TestCase.model_validate(test_case)
        test_case_name = request.node.name if request is not None else (test_case.test_case_name if test_case is not None else "monocle_test")
        token = validator.pre_test_run_setup(test_case_name, test_case.mock_tools if hasattr(test_case, "mock_tools") else None)
        test_failed:bool = False
        try:
            yield test_case
        except Exception as e:
            test_failed = True
            raise e
        finally:
            try:
                if not test_failed:
                    test_failed = validator.validate(test_case)
            finally:
                validator.post_test_cleanup(token, request.node.name, is_test_failed(request) or test_failed)
                validator._spans = []