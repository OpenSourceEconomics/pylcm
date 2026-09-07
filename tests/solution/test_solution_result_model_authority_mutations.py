"""Class-level mutation matrix for model-owned solution descriptors."""

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

import lcm.model as model_module
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy, OuterPolicyBank
from _lcm.egm.published_policy import EGMSimPolicy, NNBEGMSimPolicy
from _lcm.regime_building import processing as regime_processing
from lcm.exceptions import InvalidSimulationInputError
from lcm.solver_api import (
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    AxisRole,
)
from lcm.solvers import MSSEnvelope
from tests.simulation.test_nnbegm_split_workflow_parity import (
    _INITIAL as _ADAPTIVE_INITIAL,
)
from tests.simulation.test_nnbegm_split_workflow_parity import (
    _PARAMS as _ADAPTIVE_PARAMS,
)
from tests.simulation.test_nnbegm_split_workflow_parity import _build
from tests.test_models import n_nbegm_discrete_toy as toy
from tests.test_models.deterministic import base as deterministic_base
from tests.test_models.deterministic.dcegm_variants import (
    get_full_model,
    get_full_params,
)
from tests.test_models.n_nbegm_toy import RegimeId

_PARAMS = {"discount_factor": 0.95, "alive": {"premium": 1.0}}
_INITIAL = {
    "wealth": jnp.asarray([4.0, 15.0, 24.0]),
    "illiquid": jnp.asarray([0.0, 12.0, 20.0]),
    "age": jnp.full(3, 20.0),
    "regime_id": jnp.full(3, RegimeId.alive, dtype=jnp.int32),
}


class _EGMSimPolicySubclass(EGMSimPolicy):
    """Structurally identical payload whose exact container type is wrong."""


class _OuterPolicyBankSubclass(OuterPolicyBank):
    """Structurally identical adjuster bank whose exact type is wrong."""


class _EqualStr(str):
    """A value-equal string whose runtime type is not the built-in type."""

    __slots__ = ()


class _EqualTuple(tuple):
    """A value-equal tuple whose runtime type is not the built-in type."""

    __slots__ = ()


class _ArtifactStoreSubclass(ArtifactStore):
    """A value-equivalent artifact store with an untrusted implementation."""


class _AlternatingProjectArtifactStore(ArtifactStore):
    """A store whose repeated projections disagree about the same contents."""

    @property
    def project_calls(self) -> int:
        """Number of simulation-policy projections requested by the consumer."""
        return cast("int", getattr(self, "_project_calls", 0))

    def project(self, key: ArtifactKey) -> Mapping[int, Mapping[str, object]]:
        if key != SIMULATION_POLICY:
            return super().project(key)
        calls = self.project_calls + 1
        object.__setattr__(self, "_project_calls", calls)
        if calls % 2:
            return super().project(key)
        return MappingProxyType({})


def _must_not_run(**_kwargs: object) -> None:
    raise AssertionError("forward simulation ran before mutation rejection")


def _fixture():
    model = toy.build_model(variant="n_nbegm", n_periods=2)
    solution = model.solve(params=_PARAMS, log_level="off")
    ref = next(ref for ref in solution.replay_artifacts if ref.key == SIMULATION_POLICY)
    policy = cast("NNBEGMSimPolicy", solution.replay_artifacts[ref])
    return model, solution, ref, policy


def _with_policy(*, solution, ref: ArtifactRef, policy: object):
    """Replace one replay policy without changing any other result channel."""
    return replace(
        solution,
        replay_artifacts=ArtifactStore(dict(solution.replay_artifacts) | {ref: policy}),
    )


def _astype_egm_policy(*, policy: EGMSimPolicy, dtype: DTypeLike) -> EGMSimPolicy:
    """Co-mutate every floating leaf while keeping the payload self-consistent."""
    return replace(
        policy,
        endog_grid=policy.endog_grid.astype(dtype),
        policy=policy.policy.astype(dtype),
        value=policy.value.astype(dtype),
        marginal_utility=policy.marginal_utility.astype(dtype),
    )


def _drop_last_egm_row_axis(policy: EGMSimPolicy) -> EGMSimPolicy:
    """Drop one declared row and its matching array axis in lockstep."""
    metadata_fields = (
        "row_discrete_action_names",
        "row_passive_state_names",
        "row_discrete_state_names",
    )
    metadata_field = next(field for field in metadata_fields if getattr(policy, field))
    updates = {
        metadata_field: getattr(policy, metadata_field)[:-1],
    }
    for field in ("endog_grid", "policy", "value", "marginal_utility"):
        array = getattr(policy, field)
        updates[field] = jnp.take(array, indices=0, axis=array.ndim - 2)
    return replace(policy, **updates)


def _truncate_last_egm_node(policy: EGMSimPolicy) -> EGMSimPolicy:
    """Shorten every numeric leaf without changing its declared row roles."""
    return replace(
        policy,
        endog_grid=policy.endog_grid[..., :-1],
        policy=policy.policy[..., :-1],
        value=policy.value[..., :-1],
        marginal_utility=policy.marginal_utility[..., :-1],
    )


def _with_inverse_coefficient(
    *, policy: NNBEGMSimPolicy | NestedEGMSimPolicy, coefficient: Fraction
) -> NNBEGMSimPolicy | NestedEGMSimPolicy:
    """Co-mutate a structurally valid payload-owned inverse certificate."""
    return replace(
        policy,
        replay_capability=replace(
            policy.replay_capability,
            inverse=replace(
                policy.replay_capability.inverse,
                coefficient=coefficient,
            ),
        ),
    )


def _as_egm_subclass(policy: EGMSimPolicy) -> EGMSimPolicy:
    """Copy an EGM payload into a type-compatible but non-exact subclass."""
    return _EGMSimPolicySubclass(
        endog_grid=policy.endog_grid,
        policy=policy.policy,
        value=policy.value,
        marginal_utility=policy.marginal_utility,
        row_discrete_state_names=policy.row_discrete_state_names,
        row_passive_state_names=policy.row_passive_state_names,
        row_discrete_action_names=policy.row_discrete_action_names,
    )


@pytest.fixture(scope="module")
def ordinary_egm_fixture():
    """A genuine ordinary-EGM result with its conservative read gate enabled."""
    get_full_model.cache_clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            regime_processing,
            "_CROSSING_COMPLETE_ENVELOPES",
            (MSSEnvelope,),
        )
        # The replay gate is construction-time state, so this must be a fresh model.
        model = get_full_model(
            solver="dcegm",
            n_periods=2,
            envelope="mss",
        )
    get_full_model.cache_clear()
    params = get_full_params(n_periods=2)
    solution = model.solve(params=params, log_level="off")
    ref = next(
        ref
        for ref in solution.replay_artifacts
        if ref.key == SIMULATION_POLICY and ref.regime == "working_life"
    )
    policy = cast("EGMSimPolicy", solution.replay_artifacts[ref])
    assert policy.row_discrete_action_names == ("labor_supply",)
    initial_conditions = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([40.0]),
        "regime_id": jnp.asarray(
            [deterministic_base.RegimeId.working_life], dtype=jnp.int32
        ),
    }
    return model, params, initial_conditions, solution, ref, policy


def _adaptive_fixture():
    model = _build("adaptive")
    solution = model.solve(params=_ADAPTIVE_PARAMS, log_level="off")
    ref = next(ref for ref in solution.replay_artifacts if ref.key == SIMULATION_POLICY)
    policy = cast("NestedEGMSimPolicy", solution.replay_artifacts[ref])
    return model, solution, ref, policy


def test_stateless_terminal_carry_uses_a_synthetic_node_axis(
    ordinary_egm_fixture,
) -> None:
    """A stateless terminal carry's two padding nodes are not model state nodes."""
    _model, _params, _initial, solution, _policy_ref, _policy = ordinary_egm_fixture
    continuation_ref = next(
        ref
        for ref in solution.metadata.artifact_descriptors
        if ref.key == EGM_CONTINUATION and ref.regime == "dead"
    )
    descriptor = solution.metadata.artifact_descriptors[continuation_ref]

    assert descriptor.state_roles == ()
    assert len(descriptor.named_axes) == 1
    axis = descriptor.named_axes[0]
    assert axis.name == "pylcm:egm:node"
    assert axis.role is AxisRole.OTHER
    assert axis.coordinates == tuple(range(axis.length))


@pytest.fixture(scope="module")
def finite_authority_fixture():
    """Share one finite producer result across the broader mutation families."""
    return _fixture()


@pytest.fixture(scope="module")
def adaptive_authority_fixture():
    """Share one nested producer result across recursive payload mutations."""
    return _adaptive_fixture()


@pytest.mark.parametrize("bad_code", [-2, -1, 2, 7, 99, 2_147_483_647])
def test_every_out_of_domain_code_is_rejected(
    *, bad_code: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, solution, ref, policy = _fixture()
    codes = policy.candidate_discrete_actions
    assert codes is not None
    malformed_policy = replace(
        policy,
        candidate_discrete_actions=codes.at[0, 0].set(jnp.int32(bad_code)),
    )
    malformed = replace(
        solution,
        replay_artifacts=ArtifactStore(
            dict(solution.replay_artifacts) | {ref: malformed_policy}
        ),
    )
    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


@pytest.mark.parametrize(
    "mutation",
    ["drop_wealth", "drop_illiquid", "reverse_axes", "swap_actions"],
)
def test_axis_and_action_identity_mutations_are_rejected(
    *, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, solution, ref, policy = _fixture()
    if mutation == "drop_wealth":
        malformed_policy = replace(
            policy,
            state_names=("illiquid",),
            candidate_inner_action=policy.candidate_inner_action[:, 0, :],
            candidate_outer_target=policy.candidate_outer_target[:, 0, :],
            candidate_value=policy.candidate_value[:, 0, :],
        )
    elif mutation == "drop_illiquid":
        malformed_policy = replace(
            policy,
            state_names=("wealth",),
            candidate_inner_action=policy.candidate_inner_action[..., 0],
            candidate_outer_target=policy.candidate_outer_target[..., 0],
            candidate_value=policy.candidate_value[..., 0],
        )
    elif mutation == "reverse_axes":
        malformed_policy = replace(
            policy,
            state_names=tuple(reversed(policy.state_names)),
            candidate_inner_action=jnp.swapaxes(policy.candidate_inner_action, 1, 2),
            candidate_outer_target=jnp.swapaxes(policy.candidate_outer_target, 1, 2),
            candidate_value=jnp.swapaxes(policy.candidate_value, 1, 2),
        )
    else:
        malformed_policy = replace(
            policy,
            inner_action_name=policy.outer_action_name,
            outer_action_name=policy.inner_action_name,
        )
    malformed = replace(
        solution,
        replay_artifacts=ArtifactStore(
            dict(solution.replay_artifacts) | {ref: malformed_policy}
        ),
    )
    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


@pytest.mark.parametrize(
    "mutation", ["drop_row", "wrong_role", "truncate_nodes", "float_dtype"]
)
def test_ordinary_egm_model_authority_rejects_self_consistent_row_mutations(
    *,
    mutation: str,
    ordinary_egm_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, params, initial_conditions, solution, ref, policy = ordinary_egm_fixture
    if mutation == "drop_row":
        malformed_policy = _drop_last_egm_row_axis(policy)
    elif mutation == "wrong_role":
        # Keep the same row name and array shape but claim that the declared
        # discrete action is a discrete state. Membership alone cannot authenticate
        # which producer role owns the axis.
        malformed_policy = replace(
            policy,
            row_discrete_state_names=policy.row_discrete_action_names,
            row_discrete_action_names=(),
        )
    elif mutation == "truncate_nodes":
        malformed_policy = _truncate_last_egm_node(policy)
    else:
        malformed_policy = _astype_egm_policy(policy=policy, dtype=jnp.float16)

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"mismatched_payload|artifact payloads cannot be detached",
    ):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=_with_policy(
                solution=solution,
                ref=ref,
                policy=malformed_policy,
            ),
            log_level="off",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "drop_both_row_axes",
        "swap_action_roles",
        "swap_state_roles",
        "wrong_resources_reader",
        "truncate_outer_bank",
        "float_dtype",
    ],
)
def test_nested_egm_authority_rejects_coherent_recursive_and_role_mutations(
    *,
    mutation: str,
    adaptive_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, solution, ref, policy = adaptive_authority_fixture
    if mutation == "drop_both_row_axes":
        keeper = _drop_last_egm_row_axis(policy.keeper)
        adjuster_policies = _drop_last_egm_row_axis(policy.adjuster.policies)
        malformed_policy = replace(
            policy,
            keeper=keeper,
            adjuster=replace(policy.adjuster, policies=adjuster_policies),
        )
    elif mutation == "swap_action_roles":
        malformed_policy = replace(
            policy,
            inner_action_name=policy.outer_action_name,
            outer_action_name=policy.inner_action_name,
        )
    elif mutation == "swap_state_roles":
        malformed_policy = replace(
            policy,
            liquid_state_name=policy.outer_state_name,
            outer_state_name=policy.liquid_state_name,
        )
    elif mutation == "wrong_resources_reader":
        assert policy.resources_target_name != policy.outer_post_decision_name
        malformed_policy = replace(
            policy,
            resources_target_name=policy.outer_post_decision_name,
        )
    elif mutation == "truncate_outer_bank":
        assert policy.adjuster.outer_nodes.shape[0] > 1
        malformed_policy = replace(
            policy,
            adjuster=replace(
                policy.adjuster,
                outer_nodes=policy.adjuster.outer_nodes[:-1],
                policies=replace(
                    policy.adjuster.policies,
                    endog_grid=policy.adjuster.policies.endog_grid[:-1],
                    policy=policy.adjuster.policies.policy[:-1],
                    value=policy.adjuster.policies.value[:-1],
                    marginal_utility=(policy.adjuster.policies.marginal_utility[:-1]),
                ),
            ),
        )
    else:
        malformed_policy = replace(
            policy,
            keeper=_astype_egm_policy(policy=policy.keeper, dtype=jnp.float16),
            adjuster=replace(
                policy.adjuster,
                outer_nodes=policy.adjuster.outer_nodes.astype(jnp.float16),
                policies=_astype_egm_policy(
                    policy=policy.adjuster.policies,
                    dtype=jnp.float16,
                ),
            ),
        )

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"mismatched_payload|artifact payloads cannot be detached",
    ):
        model.simulate(
            params=_ADAPTIVE_PARAMS,
            initial_conditions=dict(_ADAPTIVE_INITIAL),
            solution=_with_policy(
                solution=solution,
                ref=ref,
                policy=malformed_policy,
            ),
            log_level="off",
        )


def test_nested_egm_authority_rejects_changed_adaptive_outer_coordinate(
    *,
    adaptive_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The payload cannot move one adaptive node while retaining its policy bank."""
    model, solution, ref, policy = adaptive_authority_fixture
    nodes = policy.adjuster.outer_nodes
    assert nodes.shape[0] >= 3
    position = nodes.shape[0] // 2
    replacement = nodes[position] + (nodes[position + 1] - nodes[position]) / 4
    malformed_nodes = nodes.at[position].set(replacement)
    assert not bool(jnp.array_equal(malformed_nodes, nodes))
    assert malformed_nodes.shape == nodes.shape
    assert malformed_nodes.dtype == nodes.dtype
    assert bool(jnp.all(jnp.diff(malformed_nodes) > 0))
    malformed_policy = replace(
        policy,
        adjuster=replace(policy.adjuster, outer_nodes=malformed_nodes),
    )

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"mismatched_payload|artifact payloads cannot be detached",
    ):
        model.simulate(
            params=_ADAPTIVE_PARAMS,
            initial_conditions=dict(_ADAPTIVE_INITIAL),
            solution=_with_policy(
                solution=solution,
                ref=ref,
                policy=malformed_policy,
            ),
            log_level="off",
        )


@pytest.mark.parametrize(
    "mutation",
    ["keeper_subclass", "adjuster_subclass", "adjuster_policies_subclass"],
)
def test_nested_egm_authority_requires_exact_recursive_container_types(
    *,
    mutation: str,
    adaptive_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, solution, ref, policy = adaptive_authority_fixture
    if mutation == "keeper_subclass":
        malformed_policy = replace(
            policy,
            keeper=_as_egm_subclass(policy.keeper),
        )
    elif mutation == "adjuster_subclass":
        malformed_policy = replace(
            policy,
            adjuster=_OuterPolicyBankSubclass(
                outer_nodes=policy.adjuster.outer_nodes,
                policies=policy.adjuster.policies,
            ),
        )
    else:
        malformed_policy = replace(
            policy,
            adjuster=replace(
                policy.adjuster,
                policies=_as_egm_subclass(policy.adjuster.policies),
            ),
        )

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"mismatched_payload|artifact payloads cannot be detached",
    ):
        model.simulate(
            params=_ADAPTIVE_PARAMS,
            initial_conditions=dict(_ADAPTIVE_INITIAL),
            solution=_with_policy(
                solution=solution,
                ref=ref,
                policy=malformed_policy,
            ),
            log_level="off",
        )


@pytest.mark.parametrize("coefficient", [Fraction(2), Fraction(1, 2), Fraction(-1)])
def test_finite_nnbegm_rejects_payload_owned_inverse_coefficient(
    *,
    coefficient: Fraction,
    finite_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, solution, ref, policy = finite_authority_fixture
    malformed_policy = _with_inverse_coefficient(
        policy=policy,
        coefficient=coefficient,
    )

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"mismatched_payload|artifact payloads cannot be detached",
    ):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=_with_policy(
                solution=solution,
                ref=ref,
                policy=malformed_policy,
            ),
            log_level="off",
        )


@pytest.mark.parametrize("coefficient", [Fraction(2), Fraction(1, 2), Fraction(-1)])
def test_nested_nnbegm_rejects_payload_owned_inverse_coefficient(
    *,
    coefficient: Fraction,
    adaptive_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, solution, ref, policy = adaptive_authority_fixture
    malformed_policy = _with_inverse_coefficient(
        policy=policy,
        coefficient=coefficient,
    )

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"mismatched_payload|artifact payloads cannot be detached",
    ):
        model.simulate(
            params=_ADAPTIVE_PARAMS,
            initial_conditions=dict(_ADAPTIVE_INITIAL),
            solution=_with_policy(
                solution=solution,
                ref=ref,
                policy=malformed_policy,
            ),
            log_level="off",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "legal_wrong_code",
        "permuted_code_rows",
        "reversed_outer_grid",
        "alternate_valid_keeper_count",
    ],
)
def test_finite_nnbegm_rejects_legal_but_model_wrong_candidate_metadata(
    *,
    mutation: str,
    finite_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, solution, ref, policy = finite_authority_fixture
    codes = policy.candidate_discrete_actions
    assert codes is not None
    assert policy.discrete_action_names
    action_name = policy.discrete_action_names[0]
    code_domain = tuple(
        model._regimes[ref.regime].simulation.discrete_grids[action_name].codes
    )
    assert len(code_domain) > 1

    if mutation == "legal_wrong_code":
        observed = int(codes[0, 0])
        other_legal_code = next(code for code in code_domain if code != observed)
        malformed_policy = replace(
            policy,
            candidate_discrete_actions=codes.at[0, 0].set(jnp.int32(other_legal_code)),
        )
    elif mutation == "permuted_code_rows":
        permuted = jnp.roll(codes, shift=1, axis=0)
        assert not bool(jnp.array_equal(permuted, codes))
        malformed_policy = replace(policy, candidate_discrete_actions=permuted)
    elif mutation == "reversed_outer_grid":
        malformed_policy = replace(
            policy,
            outer_grid_values=jnp.flip(policy.outer_grid_values),
        )
    else:
        n_candidates = policy.candidate_value.shape[0]
        n_outer = policy.outer_grid_values.shape[0]
        alternate = next(
            count
            for count in range(1, n_candidates)
            if count != policy.n_keeper_candidates
            and (n_candidates - count) % n_outer == 0
        )
        malformed_policy = replace(policy, n_keeper_candidates=alternate)

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"mismatched_payload|artifact payloads cannot be detached",
    ):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=_with_policy(
                solution=solution,
                ref=ref,
                policy=malformed_policy,
            ),
            log_level="off",
        )


@pytest.mark.parametrize("mutation", ["schema_version", "artifact_channel"])
def test_simulation_policy_identity_and_channel_are_model_owned(
    *,
    mutation: str,
    finite_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, solution, ref, policy = finite_authority_fixture
    replay_entries = dict(solution.replay_artifacts)
    replay_entries.pop(ref)
    if mutation == "schema_version":
        wrong_ref = ArtifactRef(
            period=ref.period,
            regime=ref.regime,
            key=ArtifactKey(
                type_id=SIMULATION_POLICY.type_id,
                schema_version=SIMULATION_POLICY.schema_version + 1,
            ),
        )
        replay_entries[wrong_ref] = policy
        malformed = replace(
            solution,
            replay_artifacts=ArtifactStore(replay_entries),
        )
        match = "schema versions"
    else:
        malformed = replace(
            solution,
            replay_artifacts=ArtifactStore(replay_entries),
            auxiliary_artifacts=ArtifactStore(
                dict(solution.auxiliary_artifacts) | {ref: policy}
            ),
        )
        match = "wrong channels"

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(InvalidSimulationInputError, match=match):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


def test_co_mutated_value_shape_and_axis_schema_is_rejected_before_forward(
    *,
    finite_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, solution, _ref, _policy = finite_authority_fixture
    coordinate = next(
        coordinate
        for coordinate, schema in solution.metadata.value_schemas.items()
        if coordinate[1] == "alive" and len(schema.axis_names) == 2
    )
    period, regime_name = coordinate
    original = solution.values[period][regime_name]
    original_schema = solution.metadata.value_schemas[coordinate]
    replacement = jnp.swapaxes(original, 0, 1)
    replacement_schema = replace(
        original_schema,
        shape=tuple(replacement.shape),
        axis_names=tuple(reversed(original_schema.axis_names)),
    )
    values = {
        outer_period: dict(regime_to_value)
        for outer_period, regime_to_value in solution.values.items()
    }
    values[period][regime_name] = replacement
    schemas = dict(solution.metadata.value_schemas)
    schemas[coordinate] = replacement_schema
    malformed = replace(
        solution,
        values=values,
        metadata=replace(solution.metadata, value_schemas=schemas),
    )

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(InvalidSimulationInputError, match=r"shape|axis_names"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        pytest.param("solver_api_version", True, id="solver-api-bool"),
        pytest.param("solver_api_version", 1.0, id="solver-api-float"),
        pytest.param("solver_api_version", np.int64(1), id="solver-api-numpy-int"),
        pytest.param("solution_schema_version", True, id="schema-bool"),
        pytest.param("solution_schema_version", 1.0, id="schema-float"),
        pytest.param(
            "solution_schema_version",
            np.int64(1),
            id="schema-numpy-int",
        ),
    ],
)
def test_solution_metadata_versions_require_exact_int(
    *,
    field: str,
    bad_value: object,
    finite_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Value equality cannot substitute for the metadata version's exact type."""
    model, solution, _ref, _policy = finite_authority_fixture
    assert bad_value == 1
    assert type(bad_value) is not int
    malformed_metadata = replace(solution.metadata)
    object.__setattr__(malformed_metadata, field, bad_value)
    malformed = replace(solution)
    object.__setattr__(malformed, "metadata", malformed_metadata)

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


@pytest.mark.parametrize("field", ["params_fingerprint", "model_instance_id"])
def test_solution_identity_strings_require_exact_str(
    *,
    field: str,
    finite_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A value-equal str subclass cannot enter identity or authority-cache checks."""
    model, solution, _ref, _policy = finite_authority_fixture
    original = getattr(solution.metadata, field)
    bad_value = _EqualStr(original)
    assert bad_value == original
    assert type(bad_value) is not str
    malformed_metadata = replace(solution.metadata)
    object.__setattr__(malformed_metadata, field, bad_value)
    malformed = replace(solution)
    object.__setattr__(malformed, "metadata", malformed_metadata)

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


@pytest.mark.parametrize(
    "bad_version",
    [
        pytest.param(True, id="bool"),
        pytest.param(1.0, id="float"),
    ],
)
def test_builtin_artifact_key_schema_version_requires_exact_int(
    *,
    bad_version: object,
    finite_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A constructor-bypassed built-in key must retain an exact integer version."""
    model, solution, ref, policy = finite_authority_fixture
    assert bad_version == SIMULATION_POLICY.schema_version
    assert type(bad_version) is not int
    bad_key = replace(SIMULATION_POLICY)
    object.__setattr__(bad_key, "schema_version", bad_version)
    bad_ref = replace(ref, key=bad_key)
    assert bad_ref == ref

    entries = dict(solution.replay_artifacts)
    entries.pop(ref)
    entries[bad_ref] = policy
    bad_store = ArtifactStore()
    object.__setattr__(bad_store, "_entries", MappingProxyType(entries))
    malformed = replace(solution)
    object.__setattr__(malformed, "replay_artifacts", bad_store)

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


@pytest.mark.parametrize(
    "field",
    [
        "retained_continuations",
        "replay_artifacts",
        "auxiliary_artifacts",
        "diagnostics",
    ],
)
def test_solution_result_requires_exact_artifact_store_types(
    *,
    field: str,
    finite_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No result channel may delegate authority checks to a store subclass."""
    model, solution, _ref, _policy = finite_authority_fixture
    original = getattr(solution, field)
    bad_store = _ArtifactStoreSubclass(dict(original))
    malformed = replace(solution)
    object.__setattr__(malformed, field, bad_store)

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


def test_alternating_project_store_is_rejected_before_projection(
    *,
    finite_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hostile store cannot change its answer between validation and replay."""
    model, solution, _ref, _policy = finite_authority_fixture
    bad_store = _AlternatingProjectArtifactStore(dict(solution.replay_artifacts))
    object.__setattr__(bad_store, "_project_calls", 0)
    malformed = replace(solution)
    object.__setattr__(malformed, "replay_artifacts", bad_store)

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )
    assert bad_store.project_calls == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "state_names_tuple_subclass",
        "inner_action_name_str_subclass",
        "keeper_count_float",
        "capability_tuple_subclass",
        "inverse_coefficient_bool",
        "inverse_coefficient_float",
        "inverse_low_bool",
    ],
)
def test_nnbegm_replay_metadata_and_capability_require_exact_runtime_types(
    *,
    mutation: str,
    finite_authority_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal hostile scalar and container types cannot authenticate replay facts."""
    model, solution, ref, policy = finite_authority_fixture
    malformed_policy = replace(policy)
    if mutation == "state_names_tuple_subclass":
        bad_value = _EqualTuple(policy.state_names)
        assert bad_value == policy.state_names
        object.__setattr__(malformed_policy, "state_names", bad_value)
    elif mutation == "inner_action_name_str_subclass":
        bad_value = _EqualStr(policy.inner_action_name)
        assert bad_value == policy.inner_action_name
        object.__setattr__(malformed_policy, "inner_action_name", bad_value)
    elif mutation == "keeper_count_float":
        bad_value = float(policy.n_keeper_candidates)
        assert bad_value == policy.n_keeper_candidates
        object.__setattr__(malformed_policy, "n_keeper_candidates", bad_value)
    elif mutation == "capability_tuple_subclass":
        capability = replace(policy.replay_capability)
        original = capability.undeclared_functions
        bad_value = _EqualTuple(original)
        assert bad_value == original
        object.__setattr__(capability, "undeclared_functions", bad_value)
        object.__setattr__(malformed_policy, "replay_capability", capability)
    else:
        capability = replace(policy.replay_capability)
        inverse = replace(capability.inverse)
        if mutation == "inverse_coefficient_bool":
            assert inverse.coefficient == Fraction(1)
            object.__setattr__(inverse, "coefficient", True)
        elif mutation == "inverse_coefficient_float":
            assert inverse.coefficient == Fraction(1)
            object.__setattr__(inverse, "coefficient", 1.0)
        else:
            assert inverse.low == 0.0
            object.__setattr__(inverse, "low", False)
        object.__setattr__(capability, "inverse", inverse)
        object.__setattr__(malformed_policy, "replay_capability", capability)

    monkeypatch.setattr(model_module, "simulate", _must_not_run)
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=_with_policy(
                solution=solution,
                ref=ref,
                policy=malformed_policy,
            ),
            log_level="off",
        )
