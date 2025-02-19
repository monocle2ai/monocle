class TestScopes:
    def config_scope_func(self, chain, message):
        result = chain.invoke(message)
        return result