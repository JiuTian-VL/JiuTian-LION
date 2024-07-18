import os

class Registry:
    root = os.path.expanduser("~")
    mapping = {
        "builder_name_mapping": {},
        "processor_name_mapping": {},
        "model_name_mapping": {},
        "evaluator_name_mapping": {}
    }
    
    @classmethod
    def register_builder(cls, name):
        r"""Register a dataset builder to registry with key 'name'

        Args:
            name: Key with which the dataset builder will be registered.
        """

        def wrap(builder_func):
            if name in cls.mapping["builder_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["builder_name_mapping"][name]
                    )
                )
            cls.mapping["builder_name_mapping"][name] = builder_func
            return builder_func

        return wrap
    
    @classmethod
    def register_evaluator(cls, name):
        r"""Register a task evaluator to registry with key 'name'

        Args:
            name: Key with which the task evaluator will be registered.
        """

        def wrap(eval_func):
            if name in cls.mapping["evaluator_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["evaluator_name_mapping"][name]
                    )
                )
            cls.mapping["evaluator_name_mapping"][name] = eval_func
            return eval_func

        return wrap

    @classmethod
    def register_model(cls, name):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the model will be registered.
        """

        def wrap(model_cls):
            from models import BaseModel

            assert issubclass(
                model_cls, BaseModel
            ), "All models must inherit BaseModel class"
            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def register_processor(cls, name):
        r"""Register a processor to registry with key 'name'

        Args:
            name: Key with which the processor will be registered.
        """

        def wrap(processor_cls):
            if name in cls.mapping["processor_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["processor_name_mapping"][name]
                    )
                )
            cls.mapping["processor_name_mapping"][name] = processor_cls
            return processor_cls

        return wrap

    @classmethod
    def get_builder_func(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)

    @classmethod
    def get_evaluator_func(cls, name):
        return cls.mapping["evaluator_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping["processor_name_mapping"].get(name, None)

    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def list_processors(cls):
        return sorted(cls.mapping["processor_name_mapping"].keys())

    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping["builder_name_mapping"].keys())

    @classmethod
    def list_evaluators(cls):
        return sorted(cls.mapping["evaluator_name_mapping"].keys())

registry = Registry()