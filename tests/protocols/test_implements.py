# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for @implements() decorator and protocol metadata."""

import warnings
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4

import pytest

from lionpride.protocols import (
    Hashable,
    Observable,
    Serializable,
    SignatureMismatchError,
    implements,
)


class TestImplementsDecorator:
    """Test @implements() decorator behavior and __protocols__ metadata."""

    def test_implements_sets_protocols_metadata_single(self):
        """@implements() should set __protocols__ attribute for single protocol."""

        @implements(Observable)
        class TestClass:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

        assert hasattr(TestClass, "__protocols__")
        assert len(TestClass.__protocols__) == 1
        # Protocol classes are in lionpride.protocols module
        assert TestClass.__protocols__[0].__name__ == "ObservableProto"

    def test_implements_sets_protocols_metadata_multiple(self):
        """@implements() should set __protocols__ for multiple protocols."""

        @implements(Observable, Serializable)
        class TestClass:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

            def to_dict(self, **kwargs):
                return {"id": str(self.id)}

        assert hasattr(TestClass, "__protocols__")
        assert len(TestClass.__protocols__) == 2
        protocol_names = {p.__name__ for p in TestClass.__protocols__}
        assert protocol_names == {"ObservableProto", "Serializable"}

    def test_implements_metadata_inherited_like_class_attributes(self):
        """@implements() metadata inherits via normal Python class attribute inheritance."""

        @implements(Observable)
        class Parent:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

        # Child doesn't use @implements
        class Child(Parent):
            pass

        # Parent has protocols
        assert hasattr(Parent, "__protocols__")
        assert len(Parent.__protocols__) == 1

        # Child DOES inherit __protocols__ via normal class attribute inheritance
        # (This is standard Python behavior - class attributes are inherited)
        assert hasattr(Child, "__protocols__")
        assert Child.__protocols__ == Parent.__protocols__

    def test_implements_raises_when_method_inherited_not_defined(self):
        """@implements() should raise TypeError when method is inherited, not defined in class body."""

        class Parent:
            def to_dict(self, **kwargs):
                return {"parent": "data"}

        # ❌ This violates @implements() semantics - method is inherited
        with pytest.raises(
            TypeError,
            match=r"WrongChild declares @implements\(Serializable\) but does not define 'to_dict' in its class body",
        ):

            @implements(Serializable)
            class WrongChild(Parent):
                pass  # to_dict inherited, not in body - should raise!

    def test_implements_allows_explicit_override(self):
        """@implements() should allow methods defined in class body (even if calling super)."""

        class Parent:
            def to_dict(self, **kwargs):
                return {"parent": "data"}

        # ✅ Correct: explicit override in class body
        @implements(Serializable)
        class CorrectChild(Parent):
            def to_dict(self, **kwargs):  # Explicit in body
                data = super().to_dict(**kwargs)
                data["child"] = "additional"
                return data

        # Should succeed without raising
        assert hasattr(CorrectChild, "__protocols__")
        assert Serializable in CorrectChild.__protocols__

    def test_implements_raises_when_property_inherited(self):
        """@implements() should raise TypeError when property is inherited, not defined in class body."""

        class Parent:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

        # ❌ Property inherited, not defined in Child
        with pytest.raises(
            TypeError,
            match=r"WrongChild declares @implements\(ObservableProto\) but does not define 'id' in its class body",
        ):

            @implements(Observable)
            class WrongChild(Parent):
                pass

    def test_implements_allows_property_in_class_body(self):
        """@implements() should allow properties defined in class body."""

        class Parent:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

        # ✅ Property explicitly defined in child
        @implements(Observable)
        class CorrectChild(Parent):
            @property
            def id(self) -> UUID:
                return self._id

        assert hasattr(CorrectChild, "__protocols__")
        assert Observable in CorrectChild.__protocols__

    def test_implements_raises_when_classmethod_inherited(self):
        """@implements() should raise TypeError when classmethod is inherited."""
        from lionpride.protocols import Deserializable

        class Parent:
            @classmethod
            def from_dict(cls, data, **kwargs):
                return cls()

        # ❌ Classmethod inherited
        with pytest.raises(
            TypeError,
            match=r"WrongChild declares @implements\(Deserializable\) but does not define 'from_dict' in its class body",
        ):

            @implements(Deserializable)
            class WrongChild(Parent):
                pass

    def test_implements_allows_classmethod_in_class_body(self):
        """@implements() should allow classmethods defined in class body."""
        from lionpride.protocols import Deserializable

        class Parent:
            @classmethod
            def from_dict(cls, data, **kwargs):
                return cls()

        # ✅ Classmethod explicitly defined
        @implements(Deserializable)
        class CorrectChild(Parent):
            @classmethod
            def from_dict(cls, data, **kwargs):
                instance = super().from_dict(data, **kwargs)
                return instance

        assert hasattr(CorrectChild, "__protocols__")
        assert Deserializable in CorrectChild.__protocols__

    def test_implements_validates_all_protocol_methods(self):
        """@implements() should validate ALL methods required by protocol."""

        # ✅ All methods defined
        @implements(Serializable)
        class Complete:
            def to_dict(self, **kwargs):
                return {}

        # ❌ Missing required method
        with pytest.raises(
            TypeError,
            match=r"Incomplete declares @implements\(Serializable\) but does not define 'to_dict' in its class body",
        ):

            @implements(Serializable)
            class Incomplete:
                pass

    def test_implements_validates_multiple_protocols(self):
        """@implements() should validate all methods for multiple protocols."""

        # ✅ All methods for both protocols defined
        @implements(Observable, Serializable)
        class CompleteMulti:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

            def to_dict(self, **kwargs):
                return {"id": str(self.id)}

        # ❌ Missing method from second protocol
        with pytest.raises(TypeError, match=r"but does not define 'to_dict' in its class body"):

            @implements(Observable, Serializable)
            class IncompleteMult:
                def __init__(self):
                    self._id = uuid4()

                @property
                def id(self) -> UUID:
                    return self._id

                # Missing to_dict!

    def test_isinstance_checks_structure_not_decorator(self):
        """isinstance() checks method presence, not @implements() metadata."""

        # Class without @implements but has method
        class CompleteClass:
            def __init__(self):
                self._id = uuid4()

            @property
            def id(self) -> UUID:
                return self._id

        # Class without @implements and missing method
        class IncompleteClass:
            pass  # Missing .id property

        incomplete = IncompleteClass()
        complete = CompleteClass()

        # isinstance() checks structure (method presence), not @implements()
        assert not isinstance(incomplete, Observable)  # Missing .id, no @implements
        assert isinstance(complete, Observable)  # Has .id despite no @implements

    def test_implements_with_hashable_protocol(self):
        """@implements() works with Hashable protocol."""

        @implements(Hashable)
        class TestClass:
            def __init__(self, value):
                self.value = value

            def __hash__(self):
                return hash(self.value)

            def __eq__(self, other):
                return isinstance(other, TestClass) and self.value == other.value

        assert hasattr(TestClass, "__protocols__")
        assert TestClass.__protocols__[0].__name__ == "Hashable"

        # Verify hashable behavior
        obj1 = TestClass(42)
        obj2 = TestClass(42)
        assert hash(obj1) == hash(obj2)
        assert obj1 == obj2
        assert len({obj1, obj2}) == 1  # Set deduplication

    def test_implements_empty_call_sets_empty_tuple(self):
        """@implements() with no protocols sets __protocols__ to empty tuple."""

        @implements()  # No protocols provided
        class EmptyClass:
            pass

        assert hasattr(EmptyClass, "__protocols__")
        assert EmptyClass.__protocols__ == ()

    def test_implements_validates_pydantic_fields_via_annotations(self):
        """@implements() should recognize Pydantic fields via __annotations__."""
        from pydantic import BaseModel

        # Pydantic field in __annotations__ satisfies Observable protocol
        @implements(Observable)
        class PydanticObservable(BaseModel):
            id: UUID  # Field annotation (no @property needed)

        # Should succeed without raising
        assert hasattr(PydanticObservable, "__protocols__")
        assert Observable in PydanticObservable.__protocols__

        # Verify it actually works
        instance = PydanticObservable(id=uuid4())
        assert isinstance(instance.id, UUID)


class TestSignatureVerification:
    """Test signature verification feature of @implements() decorator."""

    def test_matching_signature_no_warning(self):
        """Matching signatures should not produce warnings."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def do_work(self, x: int, y: str) -> bool: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(TestProtocol)
            class Impl:
                def do_work(self, x: int, y: str) -> bool:
                    return True

            # No warnings should be raised
            assert len(w) == 0

    def test_missing_param_warns_by_default(self):
        """Missing required parameter should warn by default."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def process(self, x: int, y: str) -> None: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(TestProtocol)
            class BadImpl:
                def process(self, x: int) -> None:  # Missing 'y' param
                    pass

            assert len(w) == 1
            assert "y" in str(w[0].message)
            assert "required by protocol" in str(w[0].message)

    def test_signature_check_error_mode(self):
        """signature_check='error' should raise SignatureMismatchError."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def process(self, x: int, y: str) -> None: ...

        with pytest.raises(SignatureMismatchError, match="'y'"):

            @implements(TestProtocol, signature_check="error")
            class BadImpl:
                def process(self, x: int) -> None:
                    pass

    def test_signature_check_skip_mode(self):
        """signature_check='skip' should skip signature verification."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def process(self, x: int, y: str) -> None: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(TestProtocol, signature_check="skip")
            class SkippedImpl:
                def process(self, x: int) -> None:
                    pass

            # No warnings because we skipped
            assert len(w) == 0

    def test_kwargs_accepts_protocol_params(self):
        """Implementation with **kwargs can satisfy protocol parameters."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def process(self, x: int, y: str, z: float) -> None: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(TestProtocol)
            class FlexibleImpl:
                def process(self, x: int, **kwargs) -> None:
                    pass

            # **kwargs satisfies y and z
            assert len(w) == 0

    def test_args_accepts_positional_params(self):
        """Implementation with *args can satisfy positional protocol parameters."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def process(self, x: int, y: str, z: float) -> None: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(TestProtocol)
            class ArgsImpl:
                def process(self, *args) -> None:
                    pass

            # *args satisfies all positional params
            assert len(w) == 0

    def test_extra_optional_params_allowed(self):
        """Implementation can have extra optional parameters."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def process(self, x: int) -> None: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(TestProtocol)
            class ExtendedImpl:
                def process(self, x: int, extra: str = "default") -> None:
                    pass

            # Extra optional param is fine
            assert len(w) == 0

    def test_extra_required_param_fails(self):
        """Implementation with extra required params should warn/error."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def process(self, x: int) -> None: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(TestProtocol)
            class BadImpl:
                def process(self, x: int, extra: str) -> None:  # extra is required
                    pass

            assert len(w) == 1
            assert "extra" in str(w[0].message)
            assert "implementation requires" in str(w[0].message)

    def test_protocol_with_kwargs_impl_must_have_kwargs(self):
        """If protocol has **kwargs, implementation must also accept **kwargs."""

        @runtime_checkable
        class FlexProtocol(Protocol):
            def process(self, x: int, **kwargs: Any) -> None: ...

        # Implementation without **kwargs should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(FlexProtocol)
            class StrictImpl:
                def process(self, x: int, y: str) -> None:
                    pass

            # Protocol has **kwargs, impl doesn't - should warn
            assert len(w) == 1
            assert "kwargs" in str(w[0].message)

        # Implementation WITH **kwargs is fine
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(FlexProtocol)
            class FlexImpl:
                def process(self, x: int, **kwargs: Any) -> None:
                    pass

            assert len(w) == 0

    def test_classmethod_signature_check(self):
        """Signature verification works with classmethods."""

        @runtime_checkable
        class FactoryProtocol(Protocol):
            @classmethod
            def create(cls, x: int, y: str) -> Any: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(FactoryProtocol)
            class GoodFactory:
                @classmethod
                def create(cls, x: int, y: str) -> Any:
                    return cls()

            assert len(w) == 0

        # Now test bad implementation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(FactoryProtocol)
            class BadFactory:
                @classmethod
                def create(cls, x: int) -> Any:  # Missing 'y'
                    return cls()

            assert len(w) == 1
            assert "y" in str(w[0].message)

    def test_staticmethod_signature_check(self):
        """Signature verification works with staticmethods."""

        @runtime_checkable
        class UtilProtocol(Protocol):
            @staticmethod
            def compute(x: int, y: int) -> int: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(UtilProtocol)
            class GoodUtil:
                @staticmethod
                def compute(x: int, y: int) -> int:
                    return x + y

            assert len(w) == 0

    def test_property_skips_signature_check(self):
        """Properties should skip signature verification (they have no params)."""

        @runtime_checkable
        class PropProtocol(Protocol):
            @property
            def value(self) -> int: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(PropProtocol)
            class PropImpl:
                @property
                def value(self) -> int:
                    return 42

            # No warnings - properties don't have signatures to compare
            assert len(w) == 0

    def test_multiple_signature_errors_aggregated(self):
        """Multiple signature errors should be aggregated in one warning/error."""

        @runtime_checkable
        class MultiMethodProtocol(Protocol):
            def method_a(self, x: int, y: str) -> None: ...
            def method_b(self, a: float, b: bool) -> None: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(MultiMethodProtocol)
            class BadMultiImpl:
                def method_a(self, x: int) -> None:  # Missing 'y'
                    pass

                def method_b(self, a: float) -> None:  # Missing 'b'
                    pass

            # Should have one warning with both errors
            assert len(w) == 1
            message = str(w[0].message)
            assert "method_a" in message
            assert "method_b" in message
            assert "y" in message
            assert "b" in message

    def test_serializable_protocol_real_world(self):
        """Real-world test with Serializable protocol."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(Serializable)
            class MySerializable:
                def __init__(self, data):
                    self.data = data

                def to_dict(self, **kwargs) -> dict:
                    return {"data": self.data}

            # Should pass - signature matches
            assert len(w) == 0

    def test_serializable_missing_kwargs_warns(self):
        """Serializable implementation without **kwargs should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(Serializable)
            class StrictSerializable:
                def to_dict(self) -> dict:  # Missing **kwargs
                    return {}

            # Should warn about kwargs
            assert len(w) == 1
            assert "kwargs" in str(w[0].message)

    def test_impl_more_lenient_is_ok(self):
        """Implementation can make required params optional."""

        @runtime_checkable
        class StrictProtocol(Protocol):
            def process(self, x: int, y: str) -> None: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(StrictProtocol)
            class LenientImpl:
                # y has default - more lenient than protocol
                def process(self, x: int, y: str = "default") -> None:
                    pass

            # More lenient is OK
            assert len(w) == 0

    def test_impl_tightening_optional_warns(self):
        """Implementation cannot make optional params required (tightening contract)."""

        @runtime_checkable
        class LenientProtocol(Protocol):
            def process(self, x: int, y: str = "default") -> None: ...

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(LenientProtocol)
            class StrictImpl:
                # y is required - tighter than protocol
                def process(self, x: int, y: str) -> None:
                    pass

            # Tightening optional to required should warn
            assert len(w) == 1
            assert "y" in str(w[0].message)
            assert "optional" in str(w[0].message)
            assert "requires" in str(w[0].message)

    def test_impl_tightening_optional_errors_with_flag(self):
        """signature_check='error' raises on tightening optional to required."""

        @runtime_checkable
        class LenientProtocol(Protocol):
            def process(self, x: int, y: str = "default") -> None: ...

        with pytest.raises(SignatureMismatchError, match="optional"):

            @implements(LenientProtocol, signature_check="error")
            class StrictImpl:
                def process(self, x: int, y: str) -> None:
                    pass


class TestAllowInherited:
    """Test allow_inherited parameter of @implements() decorator."""

    def test_inherited_method_rejected_by_default(self):
        """By default, inherited methods are not accepted."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def process(self, x: int) -> None: ...

        class Base:
            def process(self, x: int) -> None:
                pass

        with pytest.raises(TypeError, match="allow_inherited=True"):

            @implements(TestProtocol)
            class Child(Base):
                pass  # process is inherited, not in class body

    def test_allow_inherited_accepts_inherited_method(self):
        """allow_inherited=True accepts inherited methods."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def process(self, x: int) -> None: ...

        class Base:
            def process(self, x: int) -> None:
                pass

        # Should not raise
        @implements(TestProtocol, allow_inherited=True)
        class Child(Base):
            pass

        assert hasattr(Child, "__protocols__")

    def test_allow_inherited_still_requires_member_exists(self):
        """allow_inherited=True still requires the member to exist somewhere."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def process(self, x: int) -> None: ...

        class Base:
            pass  # No process method

        with pytest.raises(TypeError, match="not defined or inherited"):

            @implements(TestProtocol, allow_inherited=True)
            class Child(Base):
                pass

    def test_allow_inherited_with_signature_check(self):
        """Signature checking works with inherited methods."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def process(self, x: int, y: str) -> None: ...

        class Base:
            def process(self, x: int) -> None:  # Missing 'y'
                pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @implements(TestProtocol, allow_inherited=True)
            class Child(Base):
                pass

            # Should warn about missing 'y' parameter
            assert len(w) == 1
            assert "y" in str(w[0].message)

    def test_allow_inherited_with_override(self):
        """Class can override inherited method, and that's checked."""

        @runtime_checkable
        class TestProtocol(Protocol):
            def process(self, x: int) -> None: ...

        class Base:
            def process(self, x: int) -> None:
                pass

        # Should work - both inherited and overridden are valid
        @implements(TestProtocol, allow_inherited=True)
        class ChildWithOverride(Base):
            def process(self, x: int) -> None:
                pass

        assert hasattr(ChildWithOverride, "__protocols__")
