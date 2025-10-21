CONFIRMATION_BYZANTINE_THRESHOLD = 33
CONFIRMATION_SLASHING_THRESHOLD = 33
COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR = 5

@dataclass
class Store(object):
    time: uint64
    genesis_time: uint64
    justified_checkpoint: Checkpoint
    finalized_checkpoint: Checkpoint
    unrealized_justified_checkpoint: Checkpoint
    unrealized_finalized_checkpoint: Checkpoint
    proposer_boost_root: Root
    equivocating_indices: Set[ValidatorIndex]
    blocks: Dict[Root, BeaconBlock] = field(default_factory=dict)
    block_states: Dict[Root, BeaconState] = field(default_factory=dict)
    block_timeliness: Dict[Root, boolean] = field(default_factory=dict)
    checkpoint_states: Dict[Checkpoint, BeaconState] = field(default_factory=dict)
    latest_messages: Dict[ValidatorIndex, LatestMessage] = field(default_factory=dict)
    unrealized_justifications: Dict[Root, Checkpoint] = field(default_factory=dict)
    # New in [Confirmation Rule]
    confirmed_root: Root
    prev_epoch_unrealized_justified_checkpoint: Checkpoint
    prev_slot_head: Root


def get_forkchoice_store(anchor_state: BeaconState, anchor_block: BeaconBlock) -> Store:
    assert anchor_block.state_root == hash_tree_root(anchor_state)
    anchor_root = hash_tree_root(anchor_block)
    anchor_epoch = get_current_epoch(anchor_state)
    justified_checkpoint = Checkpoint(epoch=anchor_epoch, root=anchor_root)
    finalized_checkpoint = Checkpoint(epoch=anchor_epoch, root=anchor_root)
    proposer_boost_root = Root()
    return Store(
        time=uint64(anchor_state.genesis_time + SECONDS_PER_SLOT * anchor_state.slot),
        genesis_time=anchor_state.genesis_time,
        justified_checkpoint=justified_checkpoint,
        finalized_checkpoint=finalized_checkpoint,
        unrealized_justified_checkpoint=justified_checkpoint,
        unrealized_finalized_checkpoint=finalized_checkpoint,
        proposer_boost_root=proposer_boost_root,
        equivocating_indices=set(),
        blocks={anchor_root: copy(anchor_block)},
        block_states={anchor_root: copy(anchor_state)},
        checkpoint_states={justified_checkpoint: copy(anchor_state)},
        unrealized_justifications={anchor_root: justified_checkpoint},
        # New in [Confirmation Rule]
        confirmed_root=finalized_checkpoint.root,
        prev_epoch_unrealized_justified_checkpoint=justified_checkpoint,
        prev_slot_head=anchor_block,
    )


def is_first_slot_in_epoch(slot: Slot) -> bool:
    return compute_slots_since_epoch_start(slot) == 0


def is_ancestor(store: Store, root: Root, ancestor: Root):
    assert root in store.blocks
    assert ancestor in store.blocks
    
    return get_ancestor(store, root, store.blocks[ancestor].slot) == ancestor


def get_block_slot(store: Store, root: Root) -> Slot:
    assert root in store.blocks
    return store.blocks[root].slot


def get_block_epoch(store: Store, root: Root) -> Epoch:
    assert root in store.blocks
    return compute_epoch_at_slot(store.blocks[root].slot)


def get_checkpoint_for_block(store: Store, root: Root, epoch: Epoch) -> Checkpoint:
    return Checkpoint(get_checkpoint_block(store, root, epoch), epoch)


def get_attestation_score(store: Store, root: Root, checkpoint_state: BeaconState = None) -> Gwei:
    """
    Get LMD block weight without proposed boost
    TODO: reuse this method in ForkChoice get_weight() method later
    """
    if checkpoint_state is None:
        state = store.checkpoint_states[store.justified_checkpoint]
    else:
        state = checkpoint_state
    unslashed_and_active_indices = [
        i for i in get_active_validator_indices(state, get_current_epoch(state))
        if not state.validators[i].slashed
    ]
    attestation_score = Gwei(sum(
        state.validators[i].effective_balance for i in unslashed_and_active_indices
        if (i in store.latest_messages
            and i not in store.equivocating_indices
            and is_ancestor(store, store.latest_messages[i].root, root))
    ))
    return attestation_score


def get_slot_committee(store: Store, slot: Slot) -> Sequence[ValidatorIndex]:
    # Use post state of the head block as a source of shuffling.
    # It is safe because if the head is one epoch older than the current epoch,
    # this code won't be executed.
    head_state = store.block_states[get_head(store)]
    indices = []
    committees_count = get_committee_count_per_slot(head_state, compute_epoch_at_slot(slot))
    for i in range(committees_count):
        indices.append(get_beacon_committee(state, slot, CommitteeIndex(i)))
    return indices


def is_full_validator_set_covered(first_slot: Slot, last_slot: Slot) -> bool:
    """
    Returns whether the range from ``first_slot`` to ``last_slot`` (inclusive of both) includes an entire epoch
    """
    start_full_epoch = compute_epoch_at_slot(first_slot + (SLOTS_PER_EPOCH - 1))
    end_full_epoch = compute_epoch_at_slot(last_slot + 1) # exclusive

    return start_full_epoch < end_full_epoch


def adjust_committee_weight_estimate_to_ensure_safety(estimate: Gwei) -> Gwei:
    """
    Adjusts the ``estimate`` of the weight of a committee for a sequence of slots not covering a full epoch to
    ensure the safety of the confirmation rule with high probability.

    See https://gist.github.com/saltiniroberto/9ee53d29c33878d79417abb2b4468c20 for an explanation of why this is
    required.
    """
    return Gwei(estimate // 1000 * (1000 + COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR))


def estimate_committee_weight_between_slots(state: BeaconState, first_slot: Slot, last_slot: Slot) -> Gwei:
    """
    Returns the total weight of committees between ``first_slot`` and ``last_slot`` (inclusive of both).
    """
    total_active_balance = get_total_active_balance(state)

    start_epoch = compute_epoch_at_slot(first_slot)
    end_epoch = compute_epoch_at_slot(last_slot)

    if first_slot > last_slot:
        return Gwei(0)

    # If an entire epoch is covered by the range, return the total active balance
    if is_full_validator_set_covered(first_slot, last_slot):
        return total_active_balance

    if start_epoch == end_epoch:
        return total_active_balance // SLOTS_PER_EPOCH * (last_slot - first_slot + 1)
    else:
        # A range that spans an epoch boundary, but does not span any full epoch
        # needs pro-rata calculation

        # See https://gist.github.com/saltiniroberto/9ee53d29c33878d79417abb2b4468c20
        # for an explanation of the formula used below.

        # First, calculate the number of committees in the end epoch
        num_slots_in_end_epoch = compute_slots_since_epoch_start(last_slot) + 1
        # Next, calculate the number of slots remaining in the end epoch
        remaining_slots_in_end_epoch = SLOTS_PER_EPOCH - num_slots_in_end_epoch
        # Then, calculate the number of slots in the start epoch
        num_slots_in_start_epoch = SLOTS_PER_EPOCH - compute_slots_since_epoch_start(first_slot)

        end_epoch_weight_estimate = total_active_balance // SLOTS_PER_EPOCH * num_slots_in_end_epoch
        start_epoch_weight_estimate = (total_active_balance // SLOTS_PER_EPOCH // SLOTS_PER_EPOCH * 
            num_slots_in_start_epoch * remaining_slots_in_end_epoch)

        # Each committee from the end epoch only contributes a pro-rated weight
        return adjust_committee_weight_estimate_to_ensure_safety(
            Gwei(start_epoch_weight_estimate + end_epoch_weight_estimate)
        )


def get_equivocation_score(store: Store, balance_source: BeaconState, first_slot: Slot, last_slot: Slot) -> Gwei:
    committee_indices = set()
    for slot in range(first_slot, last_slot + 1):
        committee_indices.update(get_slot_committee(store, slot))

    equivocating_indices = committee_indices.intersection(store.equivocating_indices)
    return Gwei(
        sum(balance_source.validators[i].effective_balance for i in equivocating_indices)
    )


def compute_adversarial_weight(store: Store, balance_source: BeaconState, first_slot: Slot, last_slot: Slot) -> Gwei:
    maximum_weight = estimate_committee_weight_between_slots(balance_source, first_slot, last_slot)
    max_adversarial_weight = maximum_weight // 100 * CONFIRMATION_BYZANTINE_THRESHOLD

    # Discount total weight of equivocating validators
    equivocation_score = get_equivocation_score(store, balance_source, first_slot, last_slot)
    if max_adversarial_weight > equivocation_score:
        return Gwei(max_adversarial_weight - equivocation_score)
    else:
        return Gwei(0)


def get_block_support_in_slots(
        store: Store, balance_source: BeaconState, block_root: Root, first_slot: Slot, last_slot: Slot) -> Gwei:
    committees = []
    for slot in range(first_slot, last_slot + 1):
        committees.append(get_slot_committee(store, slot))
    unslashed_and_active_committee_indices = [
        i for i in get_active_validator_indices(balance_source, get_current_epoch(balance_source))
        if (i in committees and not balance_source.validators[i].slashed)
    ]
    return Gwei(sum(
        balance_source.validators[i].effective_balance for i in unslashed_and_active_committee_indices
        if (i in store.latest_messages
            and i not in store.equivocating_indices
            and store.latest_messages[i].root == block_root)
    ))


def compute_empty_slot_support_discount(store: Store, balance_source: BeaconState, block_root: Root) -> Gwei:
    block = store.blocks[block_root]
    parent_block = store.blocks[block.parent_root]
    # No empty slot
    if parent_block.slot + 1 == block.slot:
        return Gwei()

    # Discount votes supporting the parent block during empty slots
    # with exception to the adversarial fraction
    parent_support_in_empty_slots = get_block_support_in_slots(
        store, balance_source, block.parent_root, parent_block.slot + 1, block.slot - 1)
    adversarial_weight = compute_adversarial_weight(
        store, balance_source, parent_block.slot + 1, block.slot - 1)
    if parent_support_in_empty_slots > adversarial_weight:
        return parent_support_in_empty_slots - adversarial_weight
    else:
        return Gwei(0)


def get_support_discount(store: Store, balance_source: BeaconState, block_root: Root) -> Gwei:
    # Empty slot support discount
    return compute_empty_slot_support_discount(store, balance_source, block_root)


def get_proposer_score(balance_source: BeaconState) -> Gwei:
    committee_weight = get_total_active_balance(balance_source) // SLOTS_PER_EPOCH
    return (committee_weight * config.PROPOSER_SCORE_BOOST) // 100


def is_one_confirmed(store: Store, block_root: Root) -> bool:
    current_slot = get_current_slot(store)
    block = store.blocks[block_root]
    parent_block = store.blocks[block.parent_root]
    balance_source = store.checkpoint_states[store.prev_epoch_unrealized_justified_checkpoint]

    support = get_attestation_score(store, block_root, balance_source)
    proposer_score = get_proposer_score(balance_source)
    maximum_support = estimate_committee_weight_between_slots(balance_source, parent_block.slot + 1, current_slot - 1)
    support_discount = get_support_discount(store, balance_source, block_root)
    adversarial_weight = compute_adversarial_weight(store, balance_source, block.slot, current_slot - 1)

    # Returns whether the following condition is true using only integer arithmetic
    # support / maximum_support >
    #   0.5 * (1 + (proposer_score - support_discount) / maximum_support) + adversarial_weight / maximum_support
    return 2 * support + support_discount > maximum_support + proposer_score + 2 * adversarial_weight


def get_chain_roots(store: Store, ancestor_root: Root, root: Root) -> Sequence[Root]:
    """
    Returns block roots from ``ancestor_root`` exclusive till ``root`` inclusive.
    Returns empty list if ``ancestor_root`` is not an ancestor of ``root``.
    """
    ancestor_slot = get_block_slot(store, ancestor_root)
    chain_roots = [root]
    while store.blocks[root].slot > ancestor_slot:
        root = store.blocks[root].parent_root
        chain_roots.insert(0, root)

    if ancestor_root == store.blocks[chain_roots[0]].parent_root:
        return chain_roots
    else:
        return []


def is_chain_reconfirmed(store: Store, confirmed_root: Root) -> bool:
    # Check if the confirmed_root is descendant of prev_epoch_unrealized_justified_checkpoint
    if not is_ancestor(store, confirmed_root, store.prev_epoch_unrealized_justified_checkpoint.root):
        return False

    current_epoch = get_current_store_epoch(store)
    if store.prev_epoch_unrealized_justified_checkpoint.epoch + 1 >= current_epoch:
        # Exclude unrealized checkpoint block
        # as the this block will always be canonical in this case
        start_root = store.prev_epoch_unrealized_justified_checkpoint.root
    else:
        # Limit reconfirmation to the checkpoint block
        # as if it's successful, reconfirmation of the ancestors is implied
        checkpoint = get_checkpoint_for_block(store, confirmed_root, current_epoch - 1)
        start_root = store.blocks[checkpoint.root].parent_root

    # Run is_one_confirmed for each block in the confirmed chain
    chain_roots = get_chain_roots(store, start_root, confirmed_root)
    return all(is_one_confirmed(store, root) for root in chain_roots)


def get_checkpoint_weight(store: Store, target: Checkpoint) -> Gwei:
    """
    Uses LMD-GHOST votes to estimate FFG support for a checkpoint.
    """
    # No attestation with a vote for the target has yet been processed
    if target not in store.checkpoint_states:
        return Gwei(0)

    state = store.checkpoint_states[checkpoint]
    unslashed_and_active_indices = [
        i for i in get_active_validator_indices(state, get_current_epoch(state))
        if not state.validators[i].slashed
    ]
    return Gwei(sum(
        state.validators[i].effective_balance for i in unslashed_and_active_indices
        if (i in store.latest_messages
            and i not in store.equivocating_indices
            and target == get_checkpoint_for_block(
                    store, store.latest_messages[i].root, store.latest_messages[i].epoch))
    ))


def get_ffg_weight_till_slot(slot: Slot, epoch: Epoch, total_active_balance: Gwei) -> Gwei:
    if slot <= compute_start_slot_at_epoch(epoch):
        return Gwei(0)
    elif slot >= compute_start_slot_at_epoch(epoch + 1):
        return total_active_balance
    else:
        slots_passed = compute_slots_since_epoch_start(slot)
        return total_active_balance // SLOTS_PER_EPOCH * slots_passed


def compute_honest_ffg_support(store: Store, checkpoint: Checkpoint, state: BeaconState) -> Gwei:
    assert checkpoint.epoch == get_current_epoch_store(store)

    current_slot = get_current_slot(store)
    current_epoch = compute_epoch_at_slot(current_slot)
    total_active_balance = get_total_active_balance(state)

    # compute FFG support for checkpoint
    ffg_support_for_checkpoint = get_checkpoint_weight(store, checkpoint)

    # compute total FFG weight till current slot
    ffg_weight_till_now = get_ffg_weight_till_slot(current_slot, current_epoch, total_active_balance)

    # compute remaining honest FFG weight
    remaining_ffg_weight = total_active_balance - ffg_weight_till_now
    remaining_honest_ffg_weight = Gwei(remaining_ffg_weight // 100 * (100 - config.CONFIRMATION_BYZANTINE_THRESHOLD))

    # compute min honest FFG support
    min_honest_ffg_support = ffg_support_for_checkpoint - min(
        Gwei(ffg_weight_till_now // 100 * config.CONFIRMATION_BYZANTINE_THRESHOLD),
        Gwei(ffg_weight_till_now // 100 * config.CONFIRMATION_SLASHING_THRESHOLD),
        ffg_support_for_checkpoint
    )

    return Gwei(min_honest_ffg_support + remaining_honest_ffg_weight)


def get_checkpoint_state(store: Store, checkpoint: Checkpoint) -> BeaconState:
    if checkpoint in store.checkpoint_states:
        return store.checkpoint_states[checkpoint]
    elif checkpoint.epoch == compute_epoch_at_slot(store.block_states[checkpoint.root].slot):
        return store.block_states[checkpoint.root]
    else:
        base_state = copy(store.block_states[checkpoint.root])
        process_slots(base_state, compute_start_slot_at_epoch(target.epoch))
        return base_state


def will_no_conflicting_checkpoint_be_justified(store: Store, checkpoint: Checkpoint) -> bool:
    # If checkpoint is unrealized justified then no conflicting checkpoint can be justified
    if checkpoint == store.unrealized_justified_checkpoint:
        return True

    state = get_checkpoint_state(store, checkpoint)
    total_active_balance = get_total_active_balance(state)
    honest_ffg_support = compute_honest_ffg_support(store, checkpoint, state)
    return 3 * honest_ffg_support >= 1 * total_active_balance


def will_checkpoint_be_justified(store: Store, checkpoint: Checkpoint) -> bool:
    state = get_checkpoint_state(store, checkpoint)
    total_active_balance = get_total_active_balance(state)
    honest_ffg_support = compute_honest_ffg_support(store, checkpoint, state)
    return 3 * honest_ffg_support >= 2 * total_active_balance


def find_latest_confirmed_descendant(store: Store, latest_confirmed_root: Root) -> Root:
    """
    This function assumes that the ``latest_confirmed_root`` belongs to the canonical chain
    and is either from the previous or from the current epoch.
    """
    current_epoch = get_current_store_epoch(store)

    # verify the latest confirmed block is not too old 
    assert get_block_epoch(store, latest_confirmed_root) + 1 >= current_epoch

    head = get_head(store)
    confirmed_root = latest_confirmed_root

    if (get_block_epoch(store, confirmed_root) + 1 == current_epoch
        and get_voting_source(store, store.prev_slot_head).epoch + 2 >= current_epoch 
        and (is_first_slot_in_epoch(get_current_slot(store))
             or (will_no_conflicting_checkpoint_be_justified(store, get_checkpoint_for_block(store, head, current_epoch))
                 and (store.unrealized_justifications[store.prev_slot_head].epoch + 1 >= current_epoch
                      or store.unrealized_justifications[head].epoch + 1 >= current_epoch)))):
        # get suffix of the canonical chain
        canonical_roots = get_chain_roots(store, confirmed_root, head)

        # starting with the child of the latest_confirmed_root
        # move towards the head in attempt to advance confirmed block
        # and stop when the first unconfirmed descendant is encountered        
        for block_root in canonical_roots:        
            block_epoch = get_block_epoch(store, block_root)
            
            # If we reach the current epoch, we exit as this code is only for confirming blocks from the previous epoch
            if block_epoch == current_epoch:
                break
            
            # We can only rely on the previous head if it is a descendant of the block we are attempting to confirm
            if not is_ancestor(store, store.prev_slot_head, block_root):
                break
            
            if not is_one_confirmed(store, block_root):
                break
            
            confirmed_root = block_root
            
    if (is_first_slot_in_epoch(get_current_slot(store))
        or store.unrealized_justifications[head].epoch + 1 >= current_epoch):
        # get suffix of the canonical chain
        canonical_roots = get_chain_roots(store, confirmed_root, head)

        tentative_confirmed_root = confirmed_root

        for block_root in canonical_roots:
            block_epoch = get_block_epoch(store, block_root)
            tentative_confirmed_epoch = get_block_epoch(store, tentative_confirmed_root)
            
            # The following condition can only be true the first time that we advance to a block from the current epoch
            if block_epoch > tentative_confirmed_epoch:
                # To confirm blocks from the current epoch ensure that
                # current epoch checkpoint will be justified
                checkpoint = get_checkpoint_for_block(store, block_root, block_epoch)
                if not will_checkpoint_be_justified(store, checkpoint):
                    break  

            if not is_one_confirmed(store, block_root):
                break
                
            tentative_confirmed_root = block_root
            
        # the tentative_confirmed_root can only be confirmed if we can ensure that it is not going to be reorged out in either the current or next epoch.
        if (get_block_epoch(store, tentative_confirmed_root) == current_epoch
            or (get_voting_source(store, tentative_confirmed_root).epoch + 2 >= current_epoch
                and (is_first_slot_in_epoch(get_current_slot(store))
                     or will_no_conflicting_checkpoint_be_justified(store, get_checkpoint_for_block(store, head, current_epoch))))):
            confirmed_root = tentative_confirmed_root
            
    return confirmed_root        


def get_latest_confirmed(store: Store) -> Root:
    confirmed_root = store.confirmed_root
    current_epoch = get_current_store_epoch(store)

    # revert to finalized block if:
    # a) the latest confirmed block from two or more epochs ago
    # b) the latest confirmed block doesn't belong to the canonical chain
    # c) the confirmed chain starting from the previous epoch unrealized justified checkpoint
    #    cannot be re-confirmed at the beginning of the current epoch
    # 
    # either of the above conditions signifies that confirmation rule assumptions (at least synchrony) are broken
    # and already confirmed block might not be safe to use hence revert to the safest one which is the finalized block
    # this reversal trades monotonicity in favour of safety in the casey of asynchrony in the network
    head = get_head(store)
    if (get_block_epoch(store, confirmed_root) + 1 < current_epoch
        or not is_ancestor(store, head, confirmed_root)
        or (is_first_slot_in_epoch(get_current_slot(store)) and not is_chain_reconfirmed(store, confirmed_root))):
        confirmed_root = store.finalized_checkpoint.root
        
    # if we are at the beginning of the epoch and the epoch of the unrealized justified checkpoint at beginning of the last slot of
    # the previous epoch corresponds to the previous epoch, then we can confirm the block of the unrealized justified
    # checkpoint as, under synchrony, such a checkpoint is for sure now the greatest justified checkpoint in the view
    # of any honest validator and, therefore, any honest validator will keep voting for it for the entire epoch
    confirmed_block_slot = store.blocks[confirmed_root].slot
    prev_unrealized_justified_checkpoint_slot = store.blocks[store.prev_epoch_unrealized_justified_checkpoint.root].slot
    if (is_first_slot_in_epoch(get_current_slot(store))
        and store.prev_epoch_unrealized_justified_checkpoint.epoch + 1 == current_epoch 
        and confirmed_block_slot < prev_unrealized_justified_checkpoint_slot):
        confirmed_root = store.prev_epoch_unrealized_justified_checkpoint.root

    # attempt to further advance the latest confirmed block
    if get_block_epoch(store, confirmed_root) + 1 >= current_epoch:
        return find_latest_confirmed_descendant(store, confirmed_root)
    else:
        return confirmed_root


def on_tick_per_slot_after_attestations_applied(store: Store):
    # call sequence must be:
    # 1) on_tick(store) handler
    # 2) attestations from the previous slot are apllied to the store
    # 3) on_tick_per_slot_after_attestations_applied(store) is called
    store.confirmed_root = get_latest_confirmed(store)
    if is_first_slot_in_epoch(get_current_slot(store) + 1):
        store.prev_epoch_unrealized_justified_checkpoint = store.unrealized_justified_checkpoint
    store.prev_slot_head = get_head(store)
