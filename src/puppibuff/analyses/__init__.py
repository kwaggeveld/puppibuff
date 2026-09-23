import lazy_loader
                                        # Lazy import, uses `__init__.pyi``
__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)
