"""Source authorities for native declarations and wrapper input contracts.

These hashes bind the original class source, not arbitrary names. They do not
prove callback evaluation, information equivalence, or numerical allocation.
The independent checker separately validates each original payload authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from inspect import Signature
from types import CodeType, FunctionType, MappingProxyType, ModuleType, UnionType
from typing import TypeAliasType
from .records import JSONValue

NATIVE_INPUT_TYPES = frozenset({
    '_lcm.regime_building.value_adapter.ValueAdapter',
    '_lcm.output_roles.InRole',
    '_lcm.regime_building.schedules._PeriodDispatch',
    '_lcm.regime_building.schedules._Constant',
    '_lcm.regime_building.schedules._ProbabilityConstant',
    '_lcm.regime_building.transition_support._SupportedStochasticTransition',
    'lcm.transition.StochasticTransition',
})

NATIVE_SCHEDULE_TYPES = frozenset(name for name in NATIVE_INPUT_TYPES
    if name.startswith(('_lcm.regime_building.schedules.', '_lcm.regime_building.transition_support.'))
    or name == 'lcm.transition.StochasticTransition')

_PARTICIPATION_FILE = "19cc45ca193a562403f57f5d96525cf1e68641bef68abc3d4f4f6ea352d8301a"
_AUTHORITIES = {
    '_lcm.regime_building.schedules._PeriodDispatch': (
        ('e20e034a51b1280119819b4215d3e19cbae960e7f1baba2c10ea731e1b94b4f7',
         ('a6d5b3e9a861607d4e77f1b686e54ef817811db3b30dc8dbe474fd433ca6c64e',)),
        ('7342ffcf01253290e261e7b4e19cd1dab5194b70b526ab28f08f47159a62dc71',
         ('b9b26eacf800b5be94f84ba493e4f4686b9ac5b81610303a2ac38b8a1bee8d27',)),
    ),
    '_lcm.regime_building.schedules._Constant': (
        'fe6981c389240bef8b46ba9708461d971ccae11c7196c488181ae7327eeb09f3',
        ('a6d5b3e9a861607d4e77f1b686e54ef817811db3b30dc8dbe474fd433ca6c64e',),
    ),
    '_lcm.regime_building.schedules._ProbabilityConstant': (
        '6fa244c9f417ab0ad7bdf9b4d8c9d93f7dadeef8a4eb61c3f3dbe8baa36bb38d',
        ('b9b26eacf800b5be94f84ba493e4f4686b9ac5b81610303a2ac38b8a1bee8d27',),
    ),
    '_lcm.regime_building.transition_support._SupportedStochasticTransition': (
        '495ece700e7d79617a84d5de87e1d7242db8b72aeffc1aae76cf995b19f35080',
        ('40b09a4e739186e8e0cbbd7b0f249309e198b4183ed5465facc20c3d5411d690',),
    ),
    'lcm.transition.StochasticTransition': (
        ('747f4a807c304c2fc207a401d351d0bcd67804d0ecc6eaf5e9ced52676c59585',
         ('bee97101a92d339e52e930066ac63b88158ca786f3d63cbb562d129ea10c1383',)),
        ('a5abf1139bbcd62bccf43d2531ecd849532da5712519cfdebe005e3fa46970d8',
         ('12e409188007d10443b59c25db287487d24971dd1036b2d115a353d84e3df9c6',)),
    ),
    'lcm.temporal._TemporalDecorator': (
        'c73722027a1fcdf9b30823e45e5e6111f3b05aa44045505fbab2a6abae4a92ad',
        ('4d4c9125844b0e0612a28eeb20288dbcc0d2591a0a91c89f12f4528605e415a0',),
    ),
    '_lcm.regime_building.value_adapter.ValueAdapter': (
        ('bc46eb4aa3919573fa3252f4521a3c01f14ca6b5ff62e31e7c93b34f3ea21718',
         ('9a35754dd0bdb1b32a82d583f8bdefd0a253bb088e9d581dd41c47afe72ff7ba',)),
        ('46d796380a1afe65bf0c94fe9264db510394599c4a34f2adcfd949e94849c5b5',
         ('030fdfaf3842033b43ba4544c9f245c14e44ad24633f352b746100751f0f8e32',)),
    ),
    '_lcm.output_roles.InRole': (
        'debd4c4c4de6800dcdb328f4909cdecf033df81ff76e8433f048c439aeae794b',
        ('17dcfe8cc029b7160ab93846ad05d942b08f57162b7ec0eefc25105d020926a8',),
    ),
    '_lcm.utils.functools._WrappedCallable': (
        'd873f3c4cd5c0ac4254ea31e5bffba5181e4b520d0bef9be200c5667021529d9',
        ('f0bac3bd822f157e43b08b2f677aca8142fa0e60e101cfd6af5a2e69327132dd',),
    ),
    '_lcm.output_roles.OutputRole': (
        '8a528a29ed7f27c87d1bebd2de6fe42276af2cedc8d986758ef2cc8988f57e4f',
        ('17dcfe8cc029b7160ab93846ad05d942b08f57162b7ec0eefc25105d020926a8',),
    ),
    "lcm.participation.ParticipationRead": (
        "4d036ef554fa2b2e93e8b1f8da3366c701700c4e439006944674570d37e0d428",
        (_PARTICIPATION_FILE,),
    ),
    "lcm.participation.ParticipationCheck": (
        "44c8bb9ec6f04e6e31bd557fe5f58bfbbcb3bb5d8d2ccfedd0e86ba38c38fd36",
        (_PARTICIPATION_FILE,),
    ),
    "lcm.transition.Transition": (
        "46088bb83593235ee33e56c2f0b92a29ba51396e91d38467dac62058008c2245",
        (
            "587518f47dc4909c070ffd72e161cec0be134919a7d89f480976e6c1b0d1c496",
            # Specialization types changed in the same file. Transition is identical.
            "bee97101a92d339e52e930066ac63b88158ca786f3d63cbb562d129ea10c1383",
            # `StochasticTransition` converts its output and the age-specialized
            # types are constrained in the same file. Transition is identical.
            "12e409188007d10443b59c25db287487d24971dd1036b2d115a353d84e3df9c6",
        ),
    ),
}

_NATIVE_CALL_SOURCES = MappingProxyType({
    '_lcm.regime_building.value_adapter.ValueAdapter': frozenset({
        ('9a35754dd0bdb1b32a82d583f8bdefd0a253bb088e9d581dd41c47afe72ff7ba',
         '9f680db35e88630edd31b3d42d0abaebd3e82e72fa11aa07d9292a4c6dacca55'),
        ('030fdfaf3842033b43ba4544c9f245c14e44ad24633f352b746100751f0f8e32',
         '991053e32d97e7ea21bf0cb93648389d63d2cb6133e858de90074786457ed618'),
    }),
    '_lcm.output_roles.InRole': frozenset({
        ('17dcfe8cc029b7160ab93846ad05d942b08f57162b7ec0eefc25105d020926a8',
         '7a5565d7bde4bd3bdf2053a6de0ded841594fb997e7e409690becb26a50c1b4a'),
    }),
})

_CONVERTER_SOURCES = MappingProxyType({
    'as_bool': '0ecd2a5840b9b6f7d96bbe11863df469be7bfffbd7a84006ada8daa79a7c11ba',
    'as_canonical_float': 'ef6ed4b3697c23cb57c944670ecdee2982143e2517e21a891cd289001b62b2b9',
    'as_code': 'f156454cbfde4c224d7b7e6ad29d5fb05cfd44ad6578ad3246cffb34960f1b61',
})
_CONVERTER_FILE = 'ad663c3461f511b64bdbad3fc9741d1ee54e18c2c3df52555cbb6c1a2cace29e'
_TEMPORAL_FILE = '4d4c9125844b0e0612a28eeb20288dbcc0d2591a0a91c89f12f4528605e415a0'
_TEMPORAL_FACTORY = 'ce0fe5eb97d9ba8983f58a3d2fe4ec573269d0213a7b5324b86f0e6e9dab1aac'
_TEMPORAL_BODY = 'efc53b7b0fca2a6cf3e12b69b349956c0bb46537e6d80c9793931c5f39d86345'
_TEMPORAL_CODE_CONTRACT = 'source-bound-native-temporal-template/v1'

_SCHEDULE_CALLS = MappingProxyType({
    '_lcm.regime_building.schedules._PeriodDispatch': frozenset({
        ('a6d5b3e9a861607d4e77f1b686e54ef817811db3b30dc8dbe474fd433ca6c64e', '1bd55955bb84d7c82e795f10fb7af96251135a20d696fc6e1b45eda0483f5c5e'),
        ('b9b26eacf800b5be94f84ba493e4f4686b9ac5b81610303a2ac38b8a1bee8d27', 'eafb6be26a9b656e4b328b329e5f477d0583afe6e0868504616bedd39692d843'),
    }),
    '_lcm.regime_building.schedules._Constant': frozenset({
        ('a6d5b3e9a861607d4e77f1b686e54ef817811db3b30dc8dbe474fd433ca6c64e', '704df376d94dc6edb0d84f0902332f4c2f7ffd0813b85be19506ef538701ee1b'),
    }),
    '_lcm.regime_building.schedules._ProbabilityConstant': frozenset({
        ('b9b26eacf800b5be94f84ba493e4f4686b9ac5b81610303a2ac38b8a1bee8d27', 'a505af976343ae3f8570648b9c1239d616c3f98ec1118f9180f60cfc552d703e'),
    }),
    '_lcm.regime_building.transition_support._SupportedStochasticTransition': frozenset({
        ('bee97101a92d339e52e930066ac63b88158ca786f3d63cbb562d129ea10c1383', 'fc55c08981c08230b0817359225f8e0c4d3b87d4213897ce87d896dc45bc8f21'),
        ('12e409188007d10443b59c25db287487d24971dd1036b2d115a353d84e3df9c6', 'e42c57e95f9bd898a5daa6b6fbf5340a88ffc7a29765c66d577f71fd39febd31'),
    }),
    'lcm.transition.StochasticTransition': frozenset({
        ('bee97101a92d339e52e930066ac63b88158ca786f3d63cbb562d129ea10c1383', 'fc55c08981c08230b0817359225f8e0c4d3b87d4213897ce87d896dc45bc8f21'),
        ('12e409188007d10443b59c25db287487d24971dd1036b2d115a353d84e3df9c6', 'e42c57e95f9bd898a5daa6b6fbf5340a88ffc7a29765c66d577f71fd39febd31'),
    }),
})


def matches_native_call(*, name: str, authority: object, source: object,
                        file_sha256: object) -> bool:
    import hashlib
    if (not isinstance(source, str) or not isinstance(file_sha256, str)
            or not matches_authority(name=name, observed=authority)):
        return False
    body_sha256 = hashlib.sha256(source.strip().encode()).hexdigest()
    return (file_sha256, body_sha256) in _NATIVE_CALL_SOURCES.get(name, ())


def matches_output_converter(*, name: object, source: object,
                             file_sha256: object) -> bool:
    import hashlib
    return (isinstance(name, str) and isinstance(source, str) and file_sha256 == _CONVERTER_FILE
            and hashlib.sha256(source.strip().encode()).hexdigest() == _CONVERTER_SOURCES.get(name))


def matches_temporal_source(*, authority: object, source: object,
                            file_sha256: object) -> bool:
    import hashlib
    code_sha256 = authority.get('template_code_sha256') if isinstance(authority, Mapping) else None
    return (isinstance(authority, Mapping)
            and matches_authority(name='lcm.temporal._TemporalDecorator', observed=authority.get('class_source'))
            and authority.get('factory_source_sha256') == _TEMPORAL_FACTORY
            and authority.get('template_code_contract') == _TEMPORAL_CODE_CONTRACT
            and isinstance(code_sha256, str) and len(code_sha256) == 64
            and all(character in '0123456789abcdef' for character in code_sha256)
            and authority.get('actual_code_sha256') == code_sha256
            and file_sha256 == _TEMPORAL_FILE and isinstance(source, str)
            and hashlib.sha256(source.strip().encode()).hexdigest() == _TEMPORAL_BODY)


def matches_authority(*, name: str, observed: object) -> bool:
    """Bind both the captured class and its containing source file."""
    if name not in _AUTHORITIES or not isinstance(observed, dict):
        return False
    module, qualname = name.rsplit(".", 1)
    entry = _AUTHORITIES[name]
    variants = (entry,) if isinstance(entry[0], str) else entry
    return (observed.get("module") == module and observed.get("qualname") == qualname
            and any(observed.get("source_sha256") == source_sha256
                    and observed.get("file_sha256") in file_hashes
                    for source_sha256, file_hashes in variants))


def inspect_authority(cls: type[object]) -> dict[str, JSONValue] | None:
    import hashlib
    import inspect
    from pathlib import Path
    import sys
    name = cls.__module__ + "." + cls.__qualname__
    if name not in _AUTHORITIES:
        return None
    if (name in NATIVE_INPUT_TYPES or name in {
            '_lcm.utils.functools._WrappedCallable', '_lcm.output_roles.OutputRole',
            'lcm.temporal._TemporalDecorator'}):
        module = sys.modules.get(cls.__module__)
        if module is None or inspect.getattr_static(module, cls.__qualname__, None) is not cls:
            return None
    try:
        source = inspect.getsource(cls).strip()
        filename = inspect.getsourcefile(cls)
        # Git checkouts may use CRLF without changing the source authority.
        if filename is None:
            return None
        file_bytes = Path(filename).read_bytes().replace(b"\r\n", b"\n")
        file_hash = hashlib.sha256(file_bytes).hexdigest()
    except (TypeError, OSError):
        return None
    return {"module": cls.__module__, "qualname": cls.__qualname__,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "file_sha256": file_hash}


def inspect_native_input_authority(value: object, raw: object) -> dict[str, JSONValue] | None:
    """Authenticate a native forwarder's implementation, never evaluate it."""
    import inspect
    import sys
    import textwrap
    from enum import Enum
    cls = type(value)
    name = cls.__module__ + '.' + cls.__qualname__
    if name not in NATIVE_INPUT_TYPES:
        return None
    if name in NATIVE_SCHEDULE_TYPES:
        return inspect_native_schedule_authority(value, raw)
    authority = inspect_authority(cls)
    if authority is None or not matches_authority(name=name, observed=authority):
        return None
    if not inspect.isfunction(raw) or inspect.getattr_static(cls, '__call__', None) is not raw:
        return None
    filename_sha256 = authority.get('file_sha256')
    if (not isinstance(filename_sha256, str)
            or inspect_source_bound_function(raw, module_name=cls.__module__,
                 qualname=cls.__qualname__ + '.__call__', file_sha256=filename_sha256) is None):
        return None
    try:
        source = textwrap.dedent(inspect.getsource(raw)).strip()
    except (OSError, TypeError):
        return None
    if not matches_native_call(name=name, authority=authority, source=source,
                               file_sha256=authority['file_sha256']):
        return None
    import hashlib
    import marshal
    result: dict[str, JSONValue] = {
        'class_source': authority, 'call_code_contract': 'compiled-native-source-code/v1',
        'call_code_sha256': hashlib.sha256(marshal.dumps(raw.__code__)).hexdigest(),
    }
    if name == '_lcm.output_roles.InRole':
        base_module = sys.modules.get('_lcm.utils.functools')
        base = getattr(base_module, '_WrappedCallable', None)
        if not isinstance(base, type) or not cls.__bases__ or cls.__bases__[0] is not base:
            return None
        base_source = inspect_authority(base)
        if not matches_authority(name='_lcm.utils.functools._WrappedCallable', observed=base_source):
            return None
        module = sys.modules.get('_lcm.output_roles')
        role_class = getattr(module, 'OutputRole', None)
        role = inspect.getattr_static(value, 'role', None)
        if not isinstance(role_class, type) or not issubclass(role_class, Enum) or type(role) is not role_class:
            return None
        role_source = inspect_authority(role_class)
        if not matches_authority(name='_lcm.output_roles.OutputRole', observed=role_source):
            return None
        adapters = getattr(module, '_ADAPTERS', None)
        dtype_module = sys.modules.get('_lcm.dtypes')
        expected = {'FLOAT': 'as_canonical_float', 'BOOL': 'as_bool', 'CODE': 'as_code'}
        if (not isinstance(adapters, dict) or set(adapters) != set(role_class)
                or any(adapters[getattr(role_class, key)] is not getattr(dtype_module, target, None)
                       for key, target in expected.items())):
            return None
        result.update(base_source=base_source, role_source=role_source)
    return result


def _native_scalar_bytes(value: object) -> bytes | None:
    """Encode exact built-in constant types and floating-point bits."""
    import struct
    if value is None:
        return b'n'
    if value is Ellipsis:
        return b'e'
    if type(value) is bool:
        return b'b1' if value else b'b0'
    if type(value) is int:
        magnitude = abs(value)
        return b'i' + (b'-' if value < 0 else b'+') + magnitude.to_bytes((magnitude.bit_length() + 7) // 8, 'big')
    if type(value) is float:
        return b'f' + struct.pack('>d', value)
    if type(value) is complex:
        return b'c' + struct.pack('>dd', value.real, value.imag)
    if type(value) is str:
        return b's' + value.encode('utf-8', 'surrogatepass')
    if type(value) is bytes:
        return b'y' + value
    return None


def _native_constant_equal(left: object, right: object) -> bool:
    """Compare compiler constants without invoking application equality methods."""
    if type(left) is not type(right):
        return False
    if type(left) is slice:
        return all(_native_constant_equal(getattr(left, field), getattr(right, field))
                   for field in ("start", "stop", "step"))
    if isinstance(left, CodeType) and isinstance(right, CodeType):
        fields = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_nlocals',
                  'co_stacksize', 'co_flags', 'co_code', 'co_names', 'co_varnames',
                  'co_freevars', 'co_cellvars', 'co_name', 'co_qualname', 'co_firstlineno',
                  'co_linetable', 'co_exceptiontable', 'co_consts')
        return all(_native_constant_equal(getattr(left, field), getattr(right, field)) for field in fields)
    if type(left) is tuple and type(right) is tuple:
        return len(left) == len(right) and all(_native_constant_equal(a, b) for a, b in zip(left, right, strict=True))
    if type(left) is frozenset and type(right) is frozenset:
        left_keys, right_keys = tuple(map(_native_scalar_bytes, left)), tuple(map(_native_scalar_bytes, right))
        return (None not in left_keys and None not in right_keys
                and frozenset(left_keys) == frozenset(right_keys))
    key = _native_scalar_bytes(left)
    return key is not None and key == _native_scalar_bytes(right)


def inspect_source_bound_function(
    value: object, *, module_name: str, qualname: str, file_sha256: str,
) -> FunctionType | None:
    """Compare an actual function to code compiled from its pinned module text.

    Compilation executes no module, class, decorator, factory or function body.
    Only source-literal None defaults and closure-free methods/functions are
    admitted. Callers separately bind class, receiver and referenced globals.
    """
    import ast
    import hashlib
    import inspect
    from pathlib import Path
    import sys
    from .callable_identity import _beartype_forwarding

    if not isinstance(value, FunctionType):
        return None
    interface = _beartype_forwarding(value)
    authored = interface[0] if interface is not None else value
    if interface is None and '__wrapped__' in vars(value):
        return None
    module = sys.modules.get(module_name)
    filename = getattr(module, '__file__', None)
    if (module is None or not isinstance(filename, str) or not isinstance(authored, FunctionType)
            or authored.__globals__ is not vars(module) or authored.__closure__
            or authored.__code__.co_freevars or authored.__code__.co_qualname != qualname):
        return None
    try:
        source = Path(filename).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(source).hexdigest() != file_sha256:
            return None
        tree = ast.parse(source, filename=filename)
        body = tree.body
        parts = qualname.split('.')
        for name in parts[:-1]:
            classes = [node for node in body if isinstance(node, ast.ClassDef) and node.name == name]
            if len(classes) != 1:
                return None
            body = classes[0].body
        functions = [node for node in body if isinstance(node, ast.FunctionDef) and node.name == parts[-1]]
        if len(functions) != 1:
            return None
        arguments = functions[0].args
        defaults = [*arguments.defaults, *(node for node in arguments.kw_defaults if node is not None)]
        if any(not isinstance(node, ast.Constant) or node.value is not None for node in defaults):
            return None
        actual_defaults = authored.__defaults__
        if arguments.defaults:
            if type(actual_defaults) is not tuple or len(actual_defaults) != len(arguments.defaults) or any(value is not None for value in actual_defaults):
                return None
        elif actual_defaults is not None:
            return None
        expected_kw = {argument.arg for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True) if default is not None}
        actual_kw = authored.__kwdefaults__
        if expected_kw:
            if type(actual_kw) is not dict or set(actual_kw) != expected_kw or any(value is not None for value in actual_kw.values()):
                return None
        elif actual_kw is not None:
            return None
        compiled = compile(source, filename, 'exec', dont_inherit=True, optimize=sys.flags.optimize)
        pending, matches = [compiled], []
        while pending:
            code = pending.pop()
            if code.co_qualname == qualname:
                matches.append(code)
            pending.extend(value for value in code.co_consts if isinstance(value, CodeType))
        if len(matches) != 1 or not _native_constant_equal(authored.__code__, matches[0]):
            return None
        return authored
    except (OSError, TypeError, ValueError, SyntaxError, RecursionError):
        return None



def _native_array_alias_evaluator(
    alias: TypeAliasType,
    native: ModuleType,
    source: bytes,
    filename: str,
    *,
    name: str,
    dtype_name: str,
) -> FunctionType | None:
    """Compile the finite native alias template without running an annotation body."""
    import ast
    import builtins
    import inspect
    import sys
    evaluator = alias.evaluate_value
    if (type(evaluator) is not FunctionType or evaluator.__globals__ is not vars(native)
            or evaluator.__builtins__ is not vars(builtins) or evaluator.__closure__ is not None
            or evaluator.__annotate__ is not None or evaluator.__kwdefaults__ is not None
            or evaluator.__code__.co_name != name or evaluator.__code__.co_qualname != name
            or evaluator.__code__.co_freevars or evaluator.__code__.co_names != (dtype_name, 'Array')
            or evaluator.__code__.co_argcount != 1 or evaluator.__code__.co_posonlyargcount != 1
            or evaluator.__code__.co_kwonlyargcount != 0 or evaluator.__code__.co_varnames != ('.format',)
            or evaluator.__code__.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)
            or type(evaluator.__defaults__) is not tuple or len(evaluator.__defaults__) != 1
            or type(evaluator.__defaults__[0]) is not int or evaluator.__defaults__[0] != 1):
        return None
    aliases = [node for node in ast.parse(source).body if isinstance(node, ast.TypeAlias)
               and isinstance(node.name, ast.Name) and node.name.id == name]
    if len(aliases) != 1 or aliases[0].type_params:
        return None
    expression = aliases[0].value
    if (not isinstance(expression, ast.Subscript) or not isinstance(expression.value, ast.Name)
            or expression.value.id != dtype_name or not isinstance(expression.slice, ast.Tuple)
            or len(expression.slice.elts) != 2 or not isinstance(expression.slice.elts[0], ast.Name)
            or expression.slice.elts[0].id != 'Array' or not isinstance(expression.slice.elts[1], ast.Constant)
            or type(expression.slice.elts[1].value) is not str or expression.slice.elts[1].value != '...'):
        return None
    pending, matches = [compile(source, filename, 'exec', dont_inherit=True, optimize=sys.flags.optimize)], []
    while pending:
        code = pending.pop()
        if code.co_name == code.co_qualname == name:
            matches.append(code)
        pending.extend(item for item in code.co_consts if type(item) is CodeType)
    return evaluator if len(matches) == 1 and _native_constant_equal(evaluator.__code__, matches[0]) else None


def _native_cached_array_class(
    resolved: object,  # noqa: PAN001 - authenticate arbitrary cached alias metadata
    *,
    dtype: object,  # noqa: PAN001 - authenticate arbitrary dtype class bindings
    array: object,  # noqa: PAN001 - authenticate arbitrary array class bindings
    arrays: ModuleType,
    expected_dtypes: object,  # noqa: PAN001 - native dtype metadata is a tuple or an identity-only sentinel
) -> bool:
    """Check the same finite cached-array shape for native aliases of either dtype."""
    if not isinstance(dtype, type) or expected_dtypes is None:
        return False
    metaclass, base = vars(arrays).get('_MetaAbstractArray'), vars(arrays).get('AbstractArray')
    if type(metaclass) is not type or type(base) is not metaclass or type(resolved) is not metaclass:
        return False
    fields, bases, mro = (type.__getattribute__(resolved, name) for name in ('__dict__', '__bases__', '__mro__'))
    return (type(fields) is MappingProxyType and all(type(key) is str for key in fields)
            and type(bases) is tuple and len(bases) == 1 and bases[0] is base
            and type(mro) is tuple and len(mro) == 3 and mro[0] is resolved and mro[1] is base and mro[2] is object
            and set(fields) == {'dtype', 'array_type', 'dim_str', 'dtypes', 'dims', 'index_variadic', '__module__', '__doc__'}
            and fields['dtype'] is dtype and fields['array_type'] is array
            and type(fields['dim_str']) is str and fields['dim_str'] == '...'
            and fields['dtypes'] is expected_dtypes and fields['dtypes'] is vars(dtype).get('dtypes')
            and type(fields['dims']) is tuple and len(fields['dims']) == 1
            and fields['dims'][0] is vars(arrays).get('_anonymous_variadic_dim')
            and vars(arrays).get('_anonymous_variadic_dim') is not None
            and type(fields['index_variadic']) is int and fields['index_variadic'] == 0
            and type(fields['__module__']) is str and fields['__module__'] in ('jaxtyping', 'builtins')
            and fields['__doc__'] is None)
def inspect_native_first_array_return(
    wrapper: object, target: object,
) -> tuple[UnionType, UnionType] | None:
    """Expand only the authenticated native ValueND return-alias metadata."""
    import hashlib
    from pathlib import Path
    import sys
    from typing import TypeAliasType

    if (type(target) is not FunctionType or target.__module__ != '_lcm.dtypes'
            or target.__qualname__ != '_first_array'):
        return None
    dtypes, native_typing = sys.modules.get('_lcm.dtypes'), sys.modules.get('lcm.typing')
    numpy, jaxtyping = sys.modules.get('numpy'), sys.modules.get('jaxtyping')
    if (dtypes is None or native_typing is None or numpy is None or jaxtyping is None
            or vars(dtypes).get('_first_array') is not wrapper
            or inspect_source_bound_function(target, module_name='_lcm.dtypes',
                qualname='_first_array', file_sha256=_CONVERTER_FILE) is not target):
        return None
    from _lcm.solution import fingerprint as native

    alias = vars(native_typing).get('ValueND')
    if (type(alias) is not TypeAliasType or alias.__name__ != 'ValueND'
            or type(alias.__type_params__) is not tuple or alias.__type_params__
            or vars(dtypes).get('ValueND') is not alias
            or vars(native_typing).get('Array') is not native.Array
            or not isinstance(vars(native_typing).get('Shaped'), type)
            or vars(native_typing).get('Shaped') is not vars(jaxtyping).get('Shaped')
            or vars(dtypes).get('np') is not numpy):
        return None
    original_hint = target.__annotations__.get('return')
    if (type(original_hint) is not UnionType or len(original_hint.__args__) != 2
            or original_hint.__args__[0] is not vars(numpy).get('ndarray')
            or original_hint.__args__[1] is not alias):
        return None
    filename = vars(native_typing).get('__file__')
    if not isinstance(filename, str):
        return None
    try:
        source = Path(filename).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(source).hexdigest() != '92bd61b989e1259c67723f8d98e55f8e6b2a5fd4e4f42c9e017a63f80cd1c248':
            return None
        evaluator = _native_array_alias_evaluator(alias, native_typing, source, filename, name='ValueND', dtype_name='Shaped')
        if evaluator is None:
            return None
        before_code, before_defaults = evaluator.__code__, evaluator.__defaults__
        shaped = vars(native_typing)['Shaped']
        resolved = alias.__value__
        array_module = sys.modules.get('jaxtyping._array_types')
        if type(array_module) is not ModuleType or not _native_cached_array_class(
                resolved, dtype=shaped, array=native.Array, arrays=array_module,
                expected_dtypes=vars(array_module).get('_any_dtype')):
            return None
        if (alias.evaluate_value is not evaluator
                or evaluator.__code__ is not before_code or evaluator.__defaults__ is not before_defaults
                or evaluator.__kwdefaults__ is not None or evaluator.__closure__ is not None
                or vars(native_typing).get('ValueND') is not alias or vars(dtypes).get('ValueND') is not alias
                or vars(native_typing).get('Shaped') is not shaped or vars(jaxtyping).get('Shaped') is not shaped
                or vars(native_typing).get('Array') is not native.Array
                or Path(filename).read_bytes().replace(b'\r\n', b'\n') != source):
            return None
        expanded_hint = type.__or__(vars(numpy)['ndarray'], resolved)
        if not isinstance(expanded_hint, UnionType):
            return None
        return original_hint, expanded_hint
    except (OSError, TypeError, ValueError, SyntaxError, RecursionError):
        return None


def matches_native_code(*, authority: object, code_sha256: object) -> bool:
    return (isinstance(authority, Mapping)
            and authority.get('call_code_contract') == 'compiled-native-source-code/v1'
            and isinstance(code_sha256, str) and len(code_sha256) == 64
            and all(character in '0123456789abcdef' for character in code_sha256)
            and authority.get('call_code_sha256') == code_sha256)


def matches_schedule_call(*, name: str, authority: object, code_sha256: object) -> bool:
    """Bind a native schedule's original class, authored body and actual code."""
    import hashlib
    if not isinstance(authority, Mapping) or not matches_native_code(authority=authority, code_sha256=code_sha256):
        return False
    source = authority.get('authored_call_source')
    if (not matches_authority(name=name, observed=authority.get('class_source'))
            or not isinstance(source, str)
            or (authority.get('authored_file_sha256'), hashlib.sha256(source.strip().encode()).hexdigest())
                not in _SCHEDULE_CALLS.get(name, ())):
        return False
    return (name != '_lcm.regime_building.transition_support._SupportedStochasticTransition'
            or matches_authority(name='lcm.transition.StochasticTransition', observed=authority.get('base_source')))


def inspect_native_schedule_authority(value: object, raw: object) -> dict[str, JSONValue] | None:
    """Authenticate only source-bound native schedule templates, never their values."""
    import builtins
    import hashlib
    import inspect
    import marshal
    import sys
    import textwrap

    cls = type(value)
    name = cls.__module__ + '.' + cls.__qualname__
    authority = inspect_authority(cls)
    if (name not in NATIVE_SCHEDULE_TYPES or authority is None
            or not matches_authority(name=name, observed=authority)
            or not isinstance(raw, FunctionType) or inspect.getattr_static(cls, '__call__', None) is not raw):
        return None
    owner = cls
    base_source = None
    if name.endswith('._SupportedStochasticTransition'):
        module = sys.modules.get('lcm.transition')
        base = getattr(module, 'StochasticTransition', None)
        if (not isinstance(base, type) or cls.__bases__ != (base,)
                or inspect.getattr_static(base, '__call__', None) is not raw):
            return None
        base_source = inspect_authority(base)
        if not matches_authority(name='lcm.transition.StochasticTransition', observed=base_source):
            return None
        owner = base
    owner_source = base_source if base_source is not None else authority
    file_sha256 = owner_source.get('file_sha256')
    if not isinstance(file_sha256, str):
        return None
    authored = inspect_source_bound_function(raw, module_name=owner.__module__,
        qualname=owner.__qualname__ + '.__call__', file_sha256=file_sha256)
    if authored is None:
        return None
    module = sys.modules.get(owner.__module__)
    numpy = sys.modules.get('jax.numpy')
    if module is None or numpy is None:
        return None
    try:
        from .callable_identity import _primitive
    except ImportError:
        return None

    def numeric_member(namespace: object, module_name: str, member: str) -> bool:
        if namespace is None:
            return False
        record = _primitive(vars(namespace).get(member))
        return (isinstance(record, Mapping) and record.get('canonical') == module_name + '.' + member
                and record.get('implementation_seal') == 'captured-native-numeric-object-code/v1')

    if (not numeric_member(numpy, 'jax.numpy', 'asarray')
            or not numeric_member(sys.modules.get('numpy'), 'numpy', 'asarray')):
        return None
    if name != 'lcm.transition.StochasticTransition' and not name.endswith('._SupportedStochasticTransition'):
        if vars(module).get('jnp') is not numpy:
            return None
        if name.endswith('._PeriodDispatch') and not numeric_member(numpy, 'jax.numpy', 'where'):
            return None
        for key in ('zip', 'range', 'len'):
            if vars(module).get(key, getattr(builtins, key)) is not getattr(builtins, key):
                return None
        if name.endswith('._PeriodDispatch') and file_sha256 == 'b9b26eacf800b5be94f84ba493e4f4686b9ac5b81610303a2ac38b8a1bee8d27':
            if inspect_source_bound_function(vars(module).get('_read'), module_name=owner.__module__,
                    qualname='_read', file_sha256=file_sha256) is None:
                return None
    elif file_sha256 == '12e409188007d10443b59c25db287487d24971dd1036b2d115a353d84e3df9c6':
        dtypes = sys.modules.get('_lcm.dtypes')
        if dtypes is None:
            return None
        for key in ('as_canonical_float', 'name_of'):
            target = vars(module).get(key)
            if (target is not vars(dtypes).get(key)
                    or inspect_source_bound_function(target, module_name='_lcm.dtypes',
                        qualname=key, file_sha256=_CONVERTER_FILE) is None):
                return None
        if inspect_source_bound_function(vars(dtypes).get('_first_array'), module_name='_lcm.dtypes',
                qualname='_first_array', file_sha256=_CONVERTER_FILE) is None:
            return None
        if vars(dtypes).get('jnp') is not numpy or vars(dtypes).get('np') is not sys.modules.get('numpy'):
            return None
        from _lcm.solution import fingerprint as native
        if vars(dtypes).get('Array') is not native.Array:
            return None
        for key in ('float', 'isinstance', 'getattr', 'repr'):
            if vars(dtypes).get(key, getattr(builtins, key)) is not getattr(builtins, key):
                return None
    try:
        source = textwrap.dedent(inspect.getsource(authored)).strip()
    except (OSError, TypeError):
        return None
    result: dict[str, JSONValue] = {
        'class_source': authority,
        'base_source': base_source,
        'authored_call_source': source,
        'authored_file_sha256': file_sha256,
        'call_code_contract': 'compiled-native-source-code/v1',
        'call_code_sha256': hashlib.sha256(marshal.dumps(raw.__code__)).hexdigest(),
    }
    return result if matches_schedule_call(name=name, authority=result, code_sha256=result['call_code_sha256']) else None


def inspect_temporal_input_authority(
    value: object,
) -> tuple[Mapping[str, JSONValue], FunctionType, Signature] | None:
    """Recognize only the native consume template with its complete closure."""
    import builtins
    import hashlib
    import inspect
    import marshal
    from pathlib import Path
    import sys
    import textwrap
    # This is only a fast rejection; the native source/closure checks below
    # supply authority. Ordinary callbacks never inspect or hash native files.
    if (not isinstance(value, FunctionType) or value.__code__.co_name != 'consume'
            or value.__code__.co_qualname != '_TemporalDecorator.__call__.<locals>.consume'):
        return None
    module = sys.modules.get('lcm.temporal')
    cls = getattr(module, '_TemporalDecorator', None)
    if not isinstance(cls, type):
        return None
    class_source = inspect_authority(cls)
    if class_source is None or not matches_authority(name='lcm.temporal._TemporalDecorator', observed=class_source):
        return None
    try:
        # The factory is inspected only to identify its pinned nested code.
        factory = inspect.unwrap(inspect.getattr_static(cls, '__call__'))
        if not isinstance(factory, FunctionType) or factory.__globals__ is not vars(module):
            return None
        source = textwrap.dedent(inspect.getsource(factory)).strip()
        factory_path = inspect.getsourcefile(factory)
        if (factory_path is None or hashlib.sha256(Path(factory_path).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != _TEMPORAL_FILE
                or hashlib.sha256(source.encode()).hexdigest() != _TEMPORAL_FACTORY):
            return None
        pending, found, seen = [factory.__code__], [], set()
        while pending:
            code = pending.pop()
            if id(code) in seen:
                continue
            seen.add(id(code))
            if code.co_name == 'consume':
                found.append(code)
            pending.extend(item for item in code.co_consts if isinstance(item, CodeType))
        if len(found) != 1:
            return None
        template = found[0]
        actual_code = marshal.dumps(value.__code__.replace(co_filename='<native-temporal>'))
        template_code = marshal.dumps(template.replace(co_filename='<native-temporal>'))
        if (actual_code != template_code or value.__globals__ is not vars(module)
                or value.__code__.co_freevars != ('defaults', 'func', 'names', 'parameter_names', 'passes_period')):
            return None
        numpy = sys.modules.get('numpy')
        if numpy is None or value.__globals__.get('np') is not numpy:
            return None
        for name in ('dict', 'len', 'zip'):
            if value.__globals__.get(name, value.__builtins__.get(name)) is not getattr(builtins, name):
                return None
        if value.__closure__ is None:
            return None
        closure = {name: cell.cell_contents for name, cell in zip(value.__code__.co_freevars, value.__closure__, strict=True)}
        target, names, parameters = closure['func'], closure['names'], closure['parameter_names']
        time_names = vars(value).get('__lcm_time_params__')
        if (not isinstance(target, FunctionType) or vars(value).get('__wrapped__') is not target
                or type(names) is not tuple or not names or any(type(name) is not str for name in names)
                or len(set(names)) != len(names) or type(parameters) is not tuple
                or any(type(name) is not str for name in parameters)
                or type(closure['defaults']) is not tuple or closure['defaults']
                or type(closure['passes_period']) is not bool
                or type(time_names) is not tuple or any(type(name) is not str for name in time_names)
                or time_names != names):
            return None
        target_signature, signature = inspect.signature(target), inspect.signature(value)
        if type(target_signature) is not Signature or type(signature) is not Signature:
            return None
        target_names = tuple(target_signature.parameters)
        exposed = (*target_names, 'period') if 'period' not in target_names else target_names
        if (any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.default is not p.empty
                for p in target_signature.parameters.values())
                or parameters != exposed or tuple(signature.parameters) != exposed
                or any(p.kind != p.KEYWORD_ONLY or p.default is not p.empty for p in signature.parameters.values())
                or closure['passes_period'] != ('period' in target_names)
                or not set(names) <= set(target_names) or set(names) & {'age', 'period'}):
            return None
        authority: dict[str, JSONValue] = {
            'class_source': class_source, 'factory_source_sha256': _TEMPORAL_FACTORY,
            'template_code_contract': _TEMPORAL_CODE_CONTRACT,
            'template_code_sha256': hashlib.sha256(template_code).hexdigest(),
            'actual_code_sha256': hashlib.sha256(actual_code).hexdigest(),
            'temporal_names': [name for name in names if isinstance(name, str)],
        }
        return MappingProxyType(authority), target, signature
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def validate_original_types(value):
    """Refuse a modern spelling without the corresponding original source."""
    if isinstance(value, dict):
        name = value.get("type", "")
        if value.get("kind") not in ("dataclass", "object") or not isinstance(name, str):
            name = ""
        modern = name.rsplit(".", 1)[-1] in {"ParticipationRead", "ParticipationCheck"}
        modern_transition = name.rsplit(".", 1)[-1] == "Transition" and bool(value.get("fields", {}).get("participation"))
        if modern or modern_transition:
            if not matches_authority(name=name, observed=value.get("canonical_type_source")):
                raise ValueError("Current participation declaration lacks verified canonical class source: " + name)
        for child in value.values():
            validate_original_types(child)
    elif isinstance(value, list):
        for child in value:
            validate_original_types(child)


def gate_maps(data, *names):
    result = {}
    for name in names:
        values = data.get(name, {})
        if not isinstance(values, dict):
            raise ValueError("A transition participation/gate mapping is not captured")
        if result.keys() & values.keys():
            raise ValueError("A scheduled target has conflicting gate/participation declarations")
        result.update(values)
    return result

def inspect_native_floatnd_alias(value: object) -> dict[str, JSONValue] | None:
    """Authenticate only the actual native FloatND alias and its cached array class."""
    import hashlib
    from pathlib import Path
    import sys
    from typing import TypeAliasType

    native, jaxtyping = sys.modules.get('lcm.typing'), sys.modules.get('jaxtyping')
    arrays = sys.modules.get('jaxtyping._array_types')
    if (type(value) is not TypeAliasType or native is None or jaxtyping is None or arrays is None
            or vars(native).get('FloatND') is not value or value.__name__ != 'FloatND'
            or type(value.__type_params__) is not tuple or value.__type_params__):
        return None
    filename, array_filename = vars(native).get('__file__'), vars(arrays).get('__file__')
    if not isinstance(filename, str) or not isinstance(array_filename, str):
        return None
    try:
        source = Path(filename).read_bytes().replace(b'\r\n', b'\n')
        file_hash = hashlib.sha256(source).hexdigest()
        array_hash = hashlib.sha256(Path(array_filename).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if (file_hash not in {'92bd61b989e1259c67723f8d98e55f8e6b2a5fd4e4f42c9e017a63f80cd1c248',
                             '5335c3c4cdfdcc48c231bd7ba3e44285d93be331242d77404d7778f977fe6c99'}
                or array_hash != 'e63908f56c995f24cbed97d9ba820c8b5fd98bc0301dfb4095cde3ab7d5dd1e4'):
            return None
        from _lcm.solution import fingerprint as numerical
        dtype = vars(native).get('Float')
        if (dtype is not vars(jaxtyping).get('Float') or dtype is not vars(arrays).get('Float')
                or vars(native).get('Array') is not numerical.Array):
            return None
        evaluator = _native_array_alias_evaluator(value, native, source, filename, name='FloatND', dtype_name='Float')
        if evaluator is None:
            return None
        before_code, before_defaults = evaluator.__code__, evaluator.__defaults__
        resolved = value.__value__
        dtypes = vars(dtype).get('dtypes') if isinstance(dtype, type) else None
        expected_dtypes = ('float4_e2m1fn', 'float8_e4m3b11fnuz', 'float8_e4m3fn',
                           'float8_e4m3fnuz', 'float8_e5m2', 'float8_e5m2fnuz',
                           'bfloat16', 'float16', 'float32', 'float64')
        if (type(dtypes) is not tuple or any(type(item) is not str for item in dtypes) or dtypes != expected_dtypes
                or not _native_cached_array_class(resolved, dtype=dtype, array=numerical.Array, arrays=arrays, expected_dtypes=dtypes)
                or value.evaluate_value is not evaluator or evaluator.__code__ is not before_code or evaluator.__defaults__ is not before_defaults
                or evaluator.__kwdefaults__ is not None or evaluator.__closure__ is not None
                or vars(native).get('FloatND') is not value or vars(native).get('Float') is not dtype
                or vars(jaxtyping).get('Float') is not dtype or vars(native).get('Array') is not numerical.Array
                or Path(filename).read_bytes().replace(b'\r\n', b'\n') != source):
            return None
        return {'contract': 'source-bound-native-floatnd-alias/v1', 'alias': 'lcm.typing.FloatND',
                'typing_file_sha256': file_hash, 'array_module_sha256': array_hash,
                'array_type': 'jax.Array', 'dtype': 'jaxtyping.Float', 'shape': '...'}
    except (OSError, TypeError, ValueError, AttributeError, SyntaxError, RecursionError):
        return None


def matches_native_floatnd_alias(observed: object) -> bool:
    """Independently replay native alias/class authority, not a producer flag."""
    import sys
    native = sys.modules.get('lcm.typing')
    expected = inspect_native_floatnd_alias(vars(native).get('FloatND')) if native is not None else None
    return (expected is not None and type(observed) is dict
            and all(type(key) is str and type(value) is str for key, value in observed.items())
            and observed == expected)
