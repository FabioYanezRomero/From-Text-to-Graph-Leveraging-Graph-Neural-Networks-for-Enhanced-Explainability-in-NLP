import warnings
from dig.xgraph.method import SubgraphX

class CustomSubgraphX(SubgraphX):
    """
    Subclass of DIG's SubgraphX that allows injecting a custom value_func for marginal_contribution and explain.
    If value_func is provided, it will be used in place of the default model call.
    All other functionality is preserved.
    """
    def __init__(self, *args, value_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_value_func = value_func
        if value_func is not None:
            warnings.warn("Custom value_func injected into SubgraphX. This will override internal model calls.")

    def explain(self, *args, **kwargs):
        # If value_func is provided, inject it into kwargs (if supported by the base class)
        if self._custom_value_func is not None:
            kwargs['value_func'] = self._custom_value_func
        try:
            return super().explain(*args, **kwargs)
        except TypeError as e:
            # For legacy DIG versions, fallback to monkey-patching or error
            if 'unexpected keyword argument' in str(e) and 'value_func' in str(e):
                raise RuntimeError("Your DIG version does not support value_func in explain(). "
                                   "Please upgrade DIG or use a compatible model signature.")
            raise
