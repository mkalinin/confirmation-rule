def is_one_confirmed(store, block_root) -> bool:
    """
    Same logic as in the Confirmation Rule PR
    """
    # This method must use store.prev_slot_justified_checkpoint
    # in its computations.
    #
    # get_weight() function must also be called with
    # store.prev_slot_justified_checkpoint which requires
    # get_weight() modification.
    pass


def get_checkpoint_weight(store, checkpoint, checkpoint_state) -> Gwei:
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


def get_ffg_weight_till_slot(slot, epoch, total_active_balance) -> Gwei:
    if slot <= compute_start_slot_at_epoch(epoch):
        return Gwei(0)
    elif slot >= compute_start_slot_at_epoch(epoch + 1):
        return total_active_balance
    else:
        slots_passed = slot % SLOTS_PER_EPOCH
        return total_active_balance // SLOTS_PER_EPOCH * slots_passed


def will_current_epoch_checkpoint_be_justified(store, checkpoint) -> bool:
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


def will_checkpoint_be_justified(store, checkpoint) -> bool:
    if checkpoint == store.justified_checkpoint:
        return True

    if checkpoint == store.unrealized_justified_checkpoint:
        return True

    if checkpoint.epoch == get_current_epoch_store(store):
        return will_current_epoch_checkpoint_be_justified(store, checkpoint)

    return False


def get_canonical_roots(store, ancestor_slot) -> Sequence[Root]:
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


def find_latest_confirmed_descendant(store, latest_confirmed_root) -> Root:
    """
    This function assumes that the ``latest_confirmed_root`` belongs to the canonical chain
    and is either from the previous or from the current epoch.
    """
    current_epoch = get_current_store_epoch(store)

    # verify the latest confirmed block is not too old 
    latest_confirmed_slot = store.blocks[latest_confirmed_root].slot
    assert compute_epoch_at_slot(latest_confirmed_slot) + 1 >= current_epoch

    # retrieve suffix of the canonical chain
    # verify the latest_confirmed_root belongs to it
    canonical_roots = get_canonical_roots(store, latest_confirmed_slot)
    assert canonical_roots.pop(0) == latest_confirmed_root

    # starting with the child of the latest_confirmed_root
    # move towards the head in attempt to advance confirmed block
    # and stop when the first unconfirmed descendant is encountered
    head = get_head(store)
    confirmed_root = latest_confirmed_root
    for block_root in canonical_roots:
        block_epoch = compute_epoch_at_slot(store.blocks[block_root].slot)

        confirmed_epoch = compute_epoch_at_slot(store.blocks[confirmed_root].slot)
        if block_epoch > confirmed_epoch:
            checkpoint_root = get_checkpoint_block(store, block_root, block_epoch)
            checkpoint = Checkpoint(checkpoint_root, block_epoch)

            # To be able to confirm blocks from the current epoch ensure that:
            # 1) current epoch checkpoint will be justified
            # 2) previous epoch checkpoint is justified, although may not yet be realized.
            if (not will_checkpoint_be_justified(store, checkpoint) or 
                store.unrealized_justifications[head].epoch + 1 < current_epoch):
                break

        if is_one_confirmed(store, block_root):
            confirmed_root = block_root
        else:
            break

    return confirmed_root


def get_latest_confirmed(store) -> Root:
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
    is_confirmed_root_canonical = (confirmed_root == get_ancestor(store, head, store.blocks[confirmed_root].slot))
    if confirmed_block_epoch + 1 < current_epoch or not is_confirmed_root_canonical:
        confirmed_root = store.finalized_checkpoint.root

    # use unrealized justified root if the confirmed root is older
    # and unrealized justification is of the current epoch checkpoint
    #
    # useful in the case when the confirmation rule machinery should be restarted after a period of asynchrony
    # UJ becomes GJ in the beginning of the next epoch
    # which, assuming synchrony, every honest validator will vote for
    confirmed_block_slot = store.blocks[confirmed_root].slot
    unrealized_justified_block_slot = store.blocks[store.unrealized_justified_checkpoint.root].slot
    if (confirmed_block_slot < unrealized_justified_block_slot
        and store.unrealized_justified_checkpoint.epoch == current_epoch
        and store.prev_slot_justified_checkpoint.epoch + 1 >= current_epoch):
            confirmed_root = store.unrealized_justified_checkpoint.root

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
