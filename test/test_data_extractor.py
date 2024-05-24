def test_equivalent_lines():
    # Sample data similar to what might be in the batches and input_ids
    batches = [
        {'input_ids': [[101, 200, 0, 0], [101, 300, 400, 0], [101, 500, 600, 700]]},
        {'input_ids': [[101, 800, 900, 0], [101, 1000, 0, 0], [101, 1100, 1200, 1300]]}
    ]

    # Original line
    t_ids = [int(w) for batch in batches for ids in batch['input_ids'] for w in ids if w != 0]

    # Modified line
    token_ids = [int(token) for batch in batches for ids in batch['input_ids'] for token in ids if token != 0]

    # Print results to compare
    print("Original t_ids:", t_ids)
    print("Modified token_ids:", token_ids)

    # Check if both lists are equal
    assert t_ids == token_ids, "The lists are not equal!"

    print("Test passed: Both lists are equal.")

# Run the test function
test_equivalent_lines()


def test_ids():
    # Sample data similar to what might be in the batches and input_ids
    batches = [
        {'input_ids': [[101, 200, 0, 0], [101, 300, 400, 0], [101, 500, 600, 700]]},
        {'input_ids': [[101, 800, 900, 0], [101, 1000, 0, 0], [101, 1100, 1200, 1300]]}
    ]

    # Original line - aiming to collect index positions of non-zero tokens
    t_ids = [w_id for batch_id, batch in enumerate(batches) for sen_id, ids in
             enumerate(batch['input_ids']) for w_id, w in enumerate(ids) if w != 0]

    # Print the result to verify
    print("Original t_ids:", t_ids)

    # Expected output (manually computed or logically verified)
    expected_t_ids = [0, 1, 0, 1, 2, 0, 1, 0, 1, 2]

    # Check if the result matches the expected output
    assert t_ids == expected_t_ids, "The lists are not equal!"

    print("Test passed: The list matches the expected output.")

# Run the test function
test_ids()
