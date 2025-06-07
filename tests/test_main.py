from src.main import calculate_reimbursement


def test_calculate_reimbursement():
    """
    Tests the baseline reimbursement calculation.
    """
    # Test case 1: A standard trip
    assert calculate_reimbursement(5, 250, 150.75) == 645.0

    # Test case 2: A trip with zero values
    assert calculate_reimbursement(0, 0, 0.0) == 0.0

    # Test case 3: A short trip to check rounding and floating point math
    assert calculate_reimbursement(1, 1, 10.0) == 100.58
