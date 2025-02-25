def is_one_confirmed(store, block_root) -> bool:
    """
    Same logic as in the Confirmation Rule PR
    """
    pass


def get_checkpoint_weight(store, checkpoint, checkpoint_state) -> Gwei:
    """
    Uses LMD-GHOST votes to estimate FFG support for a checkpoint.
    """
    if get_current_slot(store) % SLOTS_PER_EPOCH == 0:
        return Gwei(0)
    
    checkpoint_weight = 0
    for validator_index, latest_message in store.latest_messages.items():
        # vote is too old
        if latest_message.epoch < checkpoint.epoch:
            continue
        
        vote_checkpoint = Checkpoint(
            root=get_checkpoint_block(store, latest_message.root, checkpoint.epoch),
            epoch=checkpoint.epoch
        )

        # vote is in favour of the checkpoint
        if checkpoint == vote_checkpoint:
            checkpoint_weight += checkpoint_state.validators[validator_index].effective_balance

    return Gwei(checkpoint_weight)


def get_remaining_honest_ffg_weight(store, total_active_balance) -> Gwei:
    """
    Computes estimated FFG weight remaining for all slots in
    [current_slot, current_epoch_last_slot] interval inclusive
    """
    current_slot = get_current_slot(store)
    if current_slot % SLOTS_PER_EPOCH == 0:
        remaining_ffg_weight = total_active_balance
    else:
        first_slot_next_epoch = compute_start_slot_at_epoch(get_current_epoch_store(store) + 1)
        slots_left = first_slot_next_epoch - current_slot
        remaining_ffg_weight = total_active_balance // SLOTS_PER_EPOCH * slots_left
    
    return Gwei(remaining_ffg_weight // 100 * (100 - config.CONFIRMATION_BYZANTINE_THRESHOLD))


def will_current_epoch_checkpoint_be_justified(store, checkpoint) -> bool:
    assert checkpoint.epoch == get_current_epoch_store(store)
    
    store_target_checkpoint_state(store, checkpoint)
    checkpoint_state = store.checkpoint_states[checkpoint]

    total_active_balance = get_total_active_balance(checkpoint_state)
    
    # compute FFG support for checkpoint
    ffg_support_for_checkpoint = get_checkpoint_weight(store, checkpoint, checkpoint_state)
    
    # compute remaining honest FFG weight
    remaining_honest_ffg_weight = get_remaining_honest_ffg_weight(store, total_active_balance)

    # compute min honest FFG support
    min_honest_ffg_support = ffg_support_for_checkpoint - min(
        Gwei(total_active_balance // 100 * config.CONFIRMATION_BYZANTINE_THRESHOLD),
        Gwei(total_active_balance // 100 * config.CONFIRMATION_SLASHING_THRESHOLD),
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
    confirmed_root = latest_confirmed_root
    for block_root in canonical_roots:
        block_epoch = compute_epoch_at_slot(store.blocks[block_root].slot)
        checkpoint = Checkpoint(
            root=get_checkpoint_block(store, block_root, block_epoch),
            epoch=block_epoch
        )
        
        if (block_epoch > compute_epoch_at_slot(latest_confirmed_slot)
            and 
                (not will_checkpoint_be_justified(store, checkpoint) or
                store.unrealized_justifications[get_head(store)].epoch + 1 >= current_epoch)
            ):
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
    is_not_going_to_be_filtered_out_during_current_epoch = (get_voting_source(store, head) + 2 >= current_epoch)
    if confirmed_block_epoch + 1 < current_epoch or not is_confirmed_root_canonical or not is_not_going_to_be_filtered_out_during_current_epoch:
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
        and store.unrealized_justified_checkpoint.epoch == current_epoch):
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
