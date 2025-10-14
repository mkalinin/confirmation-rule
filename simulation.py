EFFECTIVE_BALANCE = 32 * 1_000_000_000
SLOT_COMMITTEE_SIZE = 1024 * 64
SLOT_COMMITTEE_WEIGHT = SLOT_COMMITTEE_SIZE * EFFECTIVE_BALANCE
PROPOSER_SCORE = SLOT_COMMITTEE_WEIGHT * 2 // 5
CONFIRMATION_BYZANTINE_THRESHOLD = 25

PARTICIPATION_RATE = 100

# Empty slot
# empty_slot = 2
# block_slot_tree = [*range(0, empty_slot), empty_slot-1, *range(empty_slot, 31)]
# confirming_block_slot = empty_slot + 1
# confirming_block = block_slot_tree[confirming_block_slot]
# parent_block = block_slot_tree[confirming_block_slot-1]
# block_support_in_slot_rate = {
#     confirming_block_slot: {parent_block: 0, confirming_block: PARTICIPATION_RATE}
# }

# Late block
block_slot_tree = [*range(0, 32)]
late_block_slot = 2
confirming_block_slot = late_block_slot
confirming_block = block_slot_tree[confirming_block_slot]
parent_block = block_slot_tree[confirming_block_slot-1]
parent_block_support_rate = 75
block_support_in_slot_rate = {
    late_block_slot: {parent_block: parent_block_support_rate, confirming_block: PARTICIPATION_RATE-parent_block_support_rate}
}


def find_parent_slot(block):
    block_slot = block_slot_tree.index(block)
    assert block_slot > 0
    parent_block = block_slot_tree[block_slot - 1]

    return block_slot_tree.index(parent_block)


def get_block_support_in_slot(block, slot, participation_rate):
    rate = participation_rate
    if slot in block_support_in_slot_rate:
        if block in block_support_in_slot_rate[slot]:
            rate = block_support_in_slot_rate[slot][block]
    
    if slot < block_slot_tree.index(block):
        return 0
    else:
        return SLOT_COMMITTEE_WEIGHT // 100 * rate


def compute_block_support(block, current_slot, participation_rate):
    parent_slot = find_parent_slot(block)
    support = 0
    for slot in range(parent_slot + 1, current_slot):
        support += get_block_support_in_slot(block, slot, participation_rate)

    return support


def compute_maximum_support(block, current_slot):
    parent_slot = find_parent_slot(block)
    assert parent_slot < current_slot
    slot_count = current_slot - (parent_slot + 1)
    
    return slot_count * SLOT_COMMITTEE_WEIGHT


def compute_strict_maximum_support(block, current_slot):
    block_slot = block_slot_tree.index(block)
    assert block_slot < current_slot
    slot_count = current_slot - block_slot
    
    return slot_count * SLOT_COMMITTEE_WEIGHT


def compute_safety_indicator(block, current_slot, participation_rate):
    support = compute_block_support(block, current_slot, participation_rate)
    maximum_support = compute_maximum_support(block, current_slot)
    if maximum_support == 0:
        return 0
    else:
        return support / maximum_support


def compute_honest_parent_support(block, end_slot, current_slot, participation_rate):
    block_slot = block_slot_tree.index(block)
    parent_slot = find_parent_slot(block)
    parent = block_slot_tree[parent_slot]
    parent_support = 0
    maximum_support = 0
    adversarial_support = 0
    for slot in range(parent_slot + 1, end_slot + 1):
        parent_support += get_block_support_in_slot(parent, slot, participation_rate)
        maximum_support += SLOT_COMMITTEE_WEIGHT
        if slot < block_slot:
            adversarial_support += SLOT_COMMITTEE_WEIGHT // 100 * CONFIRMATION_BYZANTINE_THRESHOLD

    return max(0, parent_support - adversarial_support)


def get_honest_parent_support(block, current_slot, with_fix, participation_rate):
    if not with_fix:
        return 0

    block_slot = block_slot_tree.index(block)
    support_exclusive = compute_honest_parent_support(block, block_slot - 1, current_slot, participation_rate)
    support_inclusive = compute_honest_parent_support(block, block_slot, current_slot, participation_rate)
    return max(support_exclusive, support_inclusive)


def compute_safety_indicator_threshold(block, current_slot, with_fix, participation_rate):
    maximum_support = compute_maximum_support(block, current_slot)
    if with_fix:
        strict_maximum_support = compute_strict_maximum_support(block, current_slot)
    else:
        strict_maximum_support = maximum_support
    honest_parent_support = get_honest_parent_support(block, current_slot, with_fix, participation_rate)
    return (0.5 * (1 + (PROPOSER_SCORE - honest_parent_support) / maximum_support)
        + CONFIRMATION_BYZANTINE_THRESHOLD / 100 * strict_maximum_support / maximum_support)


def run(with_fix):
    print (f"CONFIRMATION_BYZANTINE_THRESHOLD={CONFIRMATION_BYZANTINE_THRESHOLD}, PARTICIPATION_RATE={PARTICIPATION_RATE}")
    for slot in range(confirming_block_slot + 1, len(block_slot_tree)):
        safety_indicator = compute_safety_indicator(confirming_block, slot, PARTICIPATION_RATE)
        maximum_support = compute_maximum_support(confirming_block, slot)
        honest_parent_support = get_honest_parent_support(confirming_block, slot, with_fix, PARTICIPATION_RATE)
        safety_indicator_threshold = compute_safety_indicator_threshold(confirming_block, slot, with_fix, PARTICIPATION_RATE)
        out = f"s + {slot-confirming_block_slot}: Q_b={safety_indicator:.4f} {'>' if safety_indicator > safety_indicator_threshold else '<'} {safety_indicator_threshold:.4f}, honest_parent_fraction={honest_parent_support/maximum_support/2:.4f}, W_b={maximum_support/SLOT_COMMITTEE_WEIGHT:.4f}*W, H_p={honest_parent_support/SLOT_COMMITTEE_WEIGHT:.4f}*W"
        if safety_indicator > safety_indicator_threshold:
            print(f"\033[32m{out}\033[0m")
        else:
            print(out)


def compute_confirmed_slot_delay(with_fix, participation_rate):
    for slot in range(confirming_block_slot + 1, len(block_slot_tree)):
        safety_indicator = compute_safety_indicator(confirming_block, slot, participation_rate)
        safety_indicator_threshold = compute_safety_indicator_threshold(confirming_block, slot, with_fix, participation_rate)
        if safety_indicator > safety_indicator_threshold:
            return slot - confirming_block_slot
    
    return len(block_slot_tree) - confirming_block_slot


def run_with_different_parent_support_rate():
    print(f"| Parent support | Delay before fix | Delay after fix | Diff |")
    print(f"|----------------|------------------|-----------------|------|")
    for parent_block_support_rate in range(10, 101, 5):
        block_support_in_slot_rate[confirming_block_slot] = {
            parent_block: parent_block_support_rate,
            confirming_block: PARTICIPATION_RATE-parent_block_support_rate
        }
        delay_wout_fix = compute_confirmed_slot_delay(False, PARTICIPATION_RATE)
        delay_with_fix = compute_confirmed_slot_delay(True, PARTICIPATION_RATE)
        print(f"| {parent_block_support_rate}% | {delay_wout_fix} | {delay_with_fix} | {delay_wout_fix - delay_with_fix} |")


def run_with_different_participation_rate():
    print(f"| Participation | Delay before fix | Delay after fix | Diff |")
    print(f"|---------------|------------------|-----------------|------|")
    for participation_rate in range(100, 89, -1):
        block_support_in_slot_rate[confirming_block_slot] = {
            parent_block: 0,
            confirming_block: participation_rate
        }
        delay_wout_fix = compute_confirmed_slot_delay(False, participation_rate)
        delay_with_fix = compute_confirmed_slot_delay(True, participation_rate)
        print(f"| {participation_rate}% | {delay_wout_fix} | {delay_with_fix} | {delay_wout_fix - delay_with_fix} |")


def run_with_different_parent_support_and_participation_rate():
    print(f"| Parent support | Participation | Delay before fix | Delay after fix | Diff |")
    print(f"|----------------|---------------|------------------|-----------------|------|")
    for parent_block_support_rate in range(10, 81, 10):
        for participation_rate in range(100, 94, -1):
            confirming_block_support_rate = (100 - parent_block_support_rate) * participation_rate // 100
            block_support_in_slot_rate[confirming_block_slot] = {
                parent_block: parent_block_support_rate * participation_rate // 100,
                confirming_block: confirming_block_support_rate
            }
            delay_wout_fix = compute_confirmed_slot_delay(False, participation_rate)
            delay_with_fix = compute_confirmed_slot_delay(True, participation_rate)
            print(f"| {parent_block_support_rate}% | {participation_rate}% | {delay_wout_fix} | {delay_with_fix} | {delay_wout_fix - delay_with_fix} |")


# run(True)
# run_with_different_parent_support_rate()
# run_with_different_participation_rate()
run_with_different_parent_support_and_participation_rate()
