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
    confirmed_root: Root  # New for confirmation rule
    prev_slot_justified_checkpoint: Checkpoint  # New for confirmation rule
    prev_slot_unrealized_justified_checkpoint: Checkpoint # New for confirmation rule
    prev_slot_head: Root # New for confirmation rule


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
        confirmed_root=finalized_checkpoint.root,  # New for confirmation rule
        prev_slot_justified_checkpoint=justified_checkpoint,  # New for confirmation rule
    )

    
def is_ancestor(store: Store, root: Root, ancestor: Root):
    assert root in store.blocks
    assert ancestor in store.blocks
    
    return get_ancestor(store, root, store.block[ancestor].slot) == ancestor


def get_block_slot(store: Store, root: Root) -> Slot:
    assert root in store.blocks
    return store.blocks[root].slot


def get_block_epoch(store: Store, root: Root) -> Epoch:
    assert root in store.blocks
    return compute_epoch_at_slot(store.blocks[root].slot)


def get_weight(store: Store, root: Root, checkpoint_state: BeaconState = None) -> Gwei:
    """
    The only modification is the new ``checkpoint_state`` param
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
    if store.proposer_boost_root == Root():
        # Return only attestation score if ``proposer_boost_root`` is not set
        return attestation_score

    # Calculate proposer score if ``proposer_boost_root`` is set
    proposer_score = Gwei(0)
    # Boost is applied if ``root`` is an ancestor of ``proposer_boost_root``
    if is_ancestor(store, store.proposer_boost_root, root):
        proposer_score = get_proposer_score(store)
    return attestation_score + proposer_score


def is_full_validator_set_covered(start_slot: Slot, end_slot: Slot) -> bool:
    """
    Returns whether the range from ``start_slot`` to ``end_slot`` (inclusive of both) includes an entire epoch
    """
    start_epoch = compute_epoch_at_slot(start_slot)
    end_epoch = compute_epoch_at_slot(end_slot)

    return (
        end_epoch > start_epoch + 1
        or (end_epoch == start_epoch + 1 and start_slot % SLOTS_PER_EPOCH == 0))


def adjust_committee_weight_estimate_to_ensure_safety(estimate: Gwei) -> Gwei:
    """
    Adjusts the ``estimate`` of the weight of a committee for a sequence of slots not covering a full epoch to
    ensure the safety of the confirmation rule with high probability.

    See https://gist.github.com/saltiniroberto/9ee53d29c33878d79417abb2b4468c20 for an explanation of why this is
    required.
    """
    return Gwei(estimate // 1000 * (1000 + COMMITTEE_WEIGHT_ESTIMATION_ADJUSTMENT_FACTOR))


def get_committee_weight_between_slots(state: BeaconState, start_slot: Slot, end_slot: Slot) -> Gwei:
    """
    Returns the total weight of committees between ``start_slot`` and ``end_slot`` (inclusive of both).
    """
    total_active_balance = get_total_active_balance(state)

    start_epoch = compute_epoch_at_slot(start_slot)
    end_epoch = compute_epoch_at_slot(end_slot)

    if start_slot > end_slot:
        return Gwei(0)

    # If an entire epoch is covered by the range, return the total active balance
    if is_full_validator_set_covered(start_slot, end_slot):
        return total_active_balance

    if start_epoch == end_epoch:
        return total_active_balance // SLOTS_PER_EPOCH * (end_slot - start_slot + 1)
    else:
        # A range that spans an epoch boundary, but does not span any full epoch
        # needs pro-rata calculation

        # See https://gist.github.com/saltiniroberto/9ee53d29c33878d79417abb2b4468c20
        # for an explanation of the formula used below.

        # First, calculate the number of committees in the end epoch
        num_slots_in_end_epoch = compute_slots_since_epoch_start(end_slot)
        # Next, calculate the number of slots remaining in the end epoch
        remaining_slots_in_end_epoch = SLOTS_PER_EPOCH - num_slots_in_end_epoch
        # Then, calculate the number of slots in the start epoch
        num_slots_in_start_epoch = SLOTS_PER_EPOCH - compute_slots_since_epoch_start(start_slot)

        end_epoch_weight_estimate = total_active_balance // SLOTS_PER_EPOCH * num_slots_in_end_epoch
        start_epoch_weight_estimate = (total_active_balance // SLOTS_PER_EPOCH // SLOTS_PER_EPOCH * 
            num_slots_in_start_epoch * remaining_slots_in_end_epoch)

        # Each committee from the end epoch only contributes a pro-rated weight
        return adjust_committee_weight_estimate_to_ensure_safety(
            Gwei(start_epoch_weight_estimate + end_epoch_weight_estimate)
        )


def is_one_confirmed(store: Store, block_root: Root) -> bool:
    current_slot = get_current_slot(store)
    block = store.blocks[block_root]
    parent_block = store.blocks[block.parent_root]
    if get_current_slot(store) % SLOTS_PER_EPOCH == 0:
        weighting_checkpoint = store.prev_slot_unrealized_justified_checkpoint
    else:
        weighting_checkpoint = store.prev_slot_justified_checkpoint
    weighting_checkpoint_state = store.checkpoint_states[weighting_checkpoint]
    support = get_weight(store, block_root, weighting_checkpoint_state)
    maximum_support = get_committee_weight_between_slots(
        weighting_checkpoint_state, Slot(parent_block.slot + 1), Slot(current_slot - 1))
    proposer_score = get_proposer_score(store)

    # Returns whether the following condition is true using only integer arithmetic
    # support / maximum_support >
    # 0.5 * (1 + proposer_score / maximum_support) + CONFIRMATION_BYZANTINE_THRESHOLD / 100

    # 2 * support > maximum_support * (1 + 2 * CONFIRMATION_BYZANTINE_THRESHOLD / 100) + proposer_score
    return (
        2 * support >
        maximum_support + maximum_support // 50 * CONFIRMATION_BYZANTINE_THRESHOLD + proposer_score
    )


def get_checkpoint_weight(store: Store, checkpoint: checkpoint_state, checkpoint_state: BeaconState) -> Gwei:
    """
    Uses LMD-GHOST votes to estimate FFG support for a checkpoint.
    """
    if get_current_slot(store) <= compute_start_slot_at_epoch(checkpoint.epoch):
        return Gwei(0)

    checkpoint_weight = 0
    for validator_index, latest_message in store.latest_messages.items():
        vote_target = Checkpoint(
            root=get_checkpoint_block(store, latest_message.root, latest_message.epoch),
            epoch=latest_message.epoch
        )
        # checkpoint matches vote's target
        if checkpoint == vote_target:
            checkpoint_weight += checkpoint_state.validators[validator_index].effective_balance

    return Gwei(checkpoint_weight)


def get_ffg_weight_till_slot(slot: Slot, epoch: Epoch, total_active_balance: Gwei) -> Gwei:
    if slot <= compute_start_slot_at_epoch(epoch):
        return Gwei(0)
    elif slot >= compute_start_slot_at_epoch(epoch + 1):
        return total_active_balance
    else:
        slots_passed = slot % SLOTS_PER_EPOCH
        return total_active_balance // SLOTS_PER_EPOCH * slots_passed


def will_current_epoch_checkpoint_be_justified(store: Store, checkpoint: Checkpoint) -> bool:
    assert checkpoint.epoch == get_current_epoch_store(store)

    current_slot = get_current_slot(store)
    current_epoch = compute_epoch_at_slot(current_slot)

    store_target_checkpoint_state(store, checkpoint)
    checkpoint_state = store.checkpoint_states[checkpoint]

    total_active_balance = get_total_active_balance(checkpoint_state)

    # compute FFG support for checkpoint
    ffg_support_for_checkpoint = get_checkpoint_weight(store, checkpoint, checkpoint_state)

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

    return 3 * (min_honest_ffg_support + remaining_honest_ffg_weight) >= 2 * total_active_balance


def will_checkpoint_be_justified(store: Store, checkpoint: Checkpoint) -> bool:
    if checkpoint == store.justified_checkpoint:
        return True

    if checkpoint == store.unrealized_justified_checkpoint:
        return True

    if checkpoint.epoch == get_current_epoch_store(store):
        return will_current_epoch_checkpoint_be_justified(store, checkpoint)

    return False


def will_no_conflicting_checkpoint_be_justified(store: Store, checkpoint: Checkpoint) -> bool:
    assert checkpoint.epoch == get_current_epoch_store(store)

    current_slot = get_current_slot(store)
    current_epoch = compute_epoch_at_slot(current_slot)

    store_target_checkpoint_state(store, checkpoint)
    checkpoint_state = store.checkpoint_states[checkpoint]

    total_active_balance = get_total_active_balance(checkpoint_state)

    # compute FFG support for checkpoint
    ffg_support_for_checkpoint = get_checkpoint_weight(store, checkpoint, checkpoint_state)

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

    return 3 * (min_honest_ffg_support + remaining_honest_ffg_weight) >= total_active_balance    


def get_canonical_roots(store: Store, ancestor_slot: Slot) -> Sequence[Root]:
    """
    Returns a suffix of the canonical chain
    including a block that is no later than the ``ancestor_slot``.
    """
    root = get_head(store)
    canonical_roots = [root]
    while store.blocks[root].slot > ancestor_slot:
        root = store.blocks[root].parent_root
        canonical_roots.insert(0, root)

    return canonical_roots


def find_latest_confirmed_descendant(store: Store, latest_confirmed_root: Root) -> Root:
    """
    This function assumes that the ``latest_confirmed_root`` belongs to the canonical chain
    and is either from the previous or from the current epoch.
    """
    current_epoch = get_current_store_epoch(store)

    # verify the latest confirmed block is not too old 
    assert compute_block_epoch(latest_confirmed_root) + 1 >= current_epoch

    head = get_head(store)
    confirmed_root = latest_confirmed_root

    if (get_block_epoch(store, confirmed_root) + 1 == current_epoch
        and get_voting_source(store, store.prev_slot_head).epoch + 2 >= current_epoch 
        and (get_current_slot(store) % SLOTS_PER_EPOCH == 0
             or (will_no_conflicting_checkpoint_be_justified(store, get_checkpoint_block(store, head, current_epoch))
                 and (store.unrealized_justifications[store.prev_slot_head].epoch + 1 >= current_epoch
                      or store.unrealized_justifications[head].epoch + 1 >= current_epoch)))):
        # retrieve suffix of the canonical chain
        # verify the latest_confirmed_root belongs to it
        canonical_roots = get_canonical_roots(store, confirmed_root)
        assert canonical_roots.pop(0) == confirmed_root
        
        # starting with the child of the latest_confirmed_root
        # move towards the head in attempt to advance confirmed block
        # and stop when the first unconfirmed descendant is encountered        
        for block_root in canonical_roots:        
            block_epoch = compute_epoch_at_slot(store.blocks[block_root].slot)
            
            # If we reach the current epoch, we exit as this code is only for confirming blocks from the previous epoch
            if block_epoch == current_epoch:
                break
            
            # We can only rely on the previous head if it is a descendant of the block we are attempting to confirm
            if not is_ancestor(store, store.prev_slot_head, block_root):
                break
            
            if not is_one_confirmed(store, block_root):
                break
            
            confirmed_root = block_root
            
    if (get_current_slot(store) % SLOTS_PER_EPOCH == 0
        or store.unrealized_justifications[head].epoch + 1 >= current_epoch):
        # retrieve suffix of the canonical chain
        # verify the latest_confirmed_root belongs to it
        canonical_roots = get_canonical_roots(store, confirmed_root)
        assert canonical_roots.pop(0) == confirmed_root

        tentative_confirmed_root = confirmed_root

        for block_root in canonical_roots:
            block_epoch = compute_epoch_at_slot(store.blocks[block_root].slot)
            tentative_confirmed_epoch = compute_epoch_at_slot(store.blocks[tentative_confirmed_root].slot)
            
            # The following condition can only be true the first time that we advance to a block from the current epoch
            if block_epoch > tentative_confirmed_epoch:
                checkpoint_root = get_checkpoint_block(store, block_root, block_epoch)
                checkpoint = Checkpoint(checkpoint_root, block_epoch)

                # To confirm blocks from the current epoch ensure that
                # current epoch checkpoint will be justified
                if not will_checkpoint_be_justified(store, checkpoint):
                    break  
            
                    
            if not is_one_confirmed(store, block_root):
                break
                
            tentative_confirmed_root = block_root
            
        # the tentative_confirmed_root can only be confirmed if we can ensure that it is not going to be reorged out in either the current or next epoch.
        if (get_block_epoch(store, tentative_confirmed_root) == current_epoch
            or (get_voting_source(store, tentative_confirmed_root).epoch + 2 >= current_epoch
                and (get_current_slot(store) % SLOTS_PER_EPOCH == 0
                     or will_no_conflicting_checkpoint_be_justified(store, get_checkpoint_block(store, head, current_epoch))))):
            confirmed_root = tentative_confirmed_root
            
    return confirmed_root        


def get_latest_confirmed(store: Store) -> Root:
    confirmed_root = store.confirmed_root
    current_epoch = get_current_store_epoch(store)

    # revert to finalized block if the latest confirmed block:
    # a) from two or more epochs ago
    # b) doesn't belong to the canonical chain
    # 
    # either of the above conditions signifies that confirmation rule assumptions (at least synchrony) are broken
    # and already confirmed block might not be safe to use hence revert to the safest one which is the finalized block
    # this reversal trades monotonicity in favour of safety in the casey of asynchrony in the network
    head = get_head(store)
    if confirmed_block_epoch + 1 < current_epoch or not is_ancestor(store, head, confirmed_root):
        confirmed_root = store.finalized_checkpoint.root
        
    # if we are at the beginning of the epoch and the epoch of the unrealized justified checkpoint at beginning of the last slot of
    # the previous epoch corresponds to the previous epoch, then we can confirm the block of the unrealized justified
    # checkpoint as, under synchrony, such a checkpoint is for sure now the greatest justified checkpoint in the view
    # of any honest validator and, therefore, any honest validator will keep voting for it for the entire epoch
    confirmed_block_slot = store.blocks[confirmed_root].slot
    prev_unrealized_justified_checkpoint_slot = store.blocks[store.prev_slot_unrealized_justified_checkpoint.root].slot

    if (get_current_slot(store) % SLOTS_PER_EPOCH == 0
        and store.prev_slot_unrealized_justified_checkpoint.epoch + 1 == current_epoch 
        and confirmed_block_slot < prev_unrealized_justified_checkpoint_slot):
        confirmed_root = store.prev_slot_unrealized_justified_checkpoint.root

    # attempt to further advance the latest confirmed block
    confirmed_block_epoch = compute_epoch_at_slot(store.blocks[confirmed_root].slot)
    if confirmed_block_epoch + 1 >= current_epoch:
        return find_latest_confirmed_descendant(store, confirmed_root)
    else:
        return confirmed_root


def on_tick_per_slot_after_attestations_applied(store: Store):
    # call sequence must be:
    # 1) on_tick(store) handler
    # 2) attestations from the previous slot are apllied to the store
    # 3) on_tick_per_slot_after_attestations_applied(store) is called
    store.confirmed_root = get_latest_confirmed(store)
    store.prev_slot_justified_checkpoint = store.justified_checkpoint
    store.prev_slot_unrealized_justified_checkpoint = store.store.unrealized_justified_checkpoint
    store.prev_slot_head = get_head(store)
