import numpy as np


def output_model(x, mileage, receipts, duration):
    """Calculates an output value based on a tiered model for mileage, receipts, and duration.

    The model consists of a base rate plus contributions from three features:
    mileage, receipts, and duration. Each feature's contribution is calculated
    using a 3-tiered system. The tiers are defined by absolute upper limits.
    Each tier has a coefficient, and an exponent parameter.

    The parameters are provided in the input list `x`.

    Let's take mileage as an example. Its contribution is the sum of three parts,
    corresponding to three tiers defined by absolute limits: [0, m01], (m01, m11], (m11, m21].

    Tier 1: `m00 * (min(mileage, m01) ** m02)`
    - `m01` is the upper limit of the first tier.
    - `min(mileage, m01)` gives the mileage value capped at the tier limit.

    Tier 2: `m10 * (min(max(0, mileage - m01), m11 - m01) ** m12)`
    - `m11` is the upper limit of the second tier.
    - `mileage - m01` is the mileage beyond the first tier.
    - `m11 - m01` is the width of the second tier.
    - `min(max(0, mileage - m01), m11 - m01)` isolates the amount of mileage within this tier.

    Tier 3: `m20 * (min(max(0, mileage - m11), m21 - m11) ** m22)`
    - `m21` is the upper limit of the third tier.
    - `mileage - m11` is the mileage beyond the second tier.
    - `m21 - m11` is the width of the third tier.
    - `min(max(0, mileage - m11), m21 - m11)` isolates the amount of mileage within this tier.

    The same tiered calculation is applied to `receipts` and `duration` with their
    respective parameters (r0*, r1*, r2* and d0*, d1*, d2*).
    """
    base_rate = x[0]

    # Tiered model parameters for mileage
    # m*1 parameters are the absolute upper limits for each tier.
    m00, m01, m02 = x[1], x[2], x[3]  # Tier 1
    m10, m11, m12 = x[4], x[5], x[6]  # Tier 2
    m20, m21, m22 = x[7], x[8], x[9]  # Tier 3

    # Tiered model parameters for receipts
    # r*1 parameters are the absolute upper limits for each tier.
    r00, r01, r02 = x[10], x[11], x[12]  # Tier 1
    r10, r11, r12 = x[13], x[14], x[15]  # Tier 2
    r20, r21, r22 = x[16], x[17], x[18]  # Tier 3

    # Tiered model parameters for duration
    # d*1 parameters are the absolute upper limits for each tier.
    d00, d01, d02 = x[19], x[20], x[21]  # Tier 1
    d10, d11, d12 = x[22], x[23], x[24]  # Tier 2
    d20, d21, d22 = x[25], x[26], x[27]  # Tier 3

    # Ensure inputs are array-like for vectorized operations
    mileage = np.asarray(mileage)
    receipts = np.asarray(receipts)
    duration = np.asarray(duration)

    out = (
        base_rate
        # Mileage contribution
        + m00 * (np.abs(np.minimum(mileage, m01)) ** m02)
        + m10 * (np.abs(np.minimum(np.maximum(0, mileage - m01), m11 - m01)) ** m12)
        + m20 * (np.abs(np.minimum(np.maximum(0, mileage - m11), m21 - m11)) ** m22)
        # Receipts contribution
        + r00 * (np.abs(np.minimum(receipts, r01)) ** r02)
        + r10 * (np.abs(np.minimum(np.maximum(0, receipts - r01), r11 - r01)) ** r12)
        + r20 * (np.abs(np.minimum(np.maximum(0, receipts - r11), r21 - r11)) ** r22)
        # Duration contribution
        + d00 * (np.abs(np.minimum(duration, d01)) ** d02)
        + d10 * (np.abs(np.minimum(np.maximum(0, duration - d01), d11 - d01)) ** d12)
        + d20 * (np.abs(np.minimum(np.maximum(0, duration - d11), d21 - d11)) ** d22)
    )
    return out
