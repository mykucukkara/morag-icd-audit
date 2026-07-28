class MoragError(Exception):
    pass

class LLMOutputParsingError(MoragError):
    pass

class OOMError(MoragError):
    pass
