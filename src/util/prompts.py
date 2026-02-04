# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, you can obtain one at the repository root in LICENSE.

# helper function to load/initalize the prompt
def load_sample_prompt(dataframe, sample_idx=0, structured_guidelines=False):
	"""
	Given a dataframe with openreviews use a sample
	from the dataset and generate a prompt that will
	be supplied to a model to generate peer review

	Parameters
	------------
	arg1 | dataframe: pl.DataFrame
		Dataframe with openreview data for papers
	arg2 | sample_idx: int[OPTIONAL]
		Sample index that serves as row number from the dataframe
	arg3 | structured_guidelines[OPTIONAL]
		Boolean flag to return a prompt with guidelines and support
		structured outputs

	Returns
	------------
		Text
	"""
	# prompt init
	prompt, input_text = None, None

	# prompt with guidelines
	if structured_guidelines:
		# prompt with rules and guidelines
		input_text = """
		<|begin_of_text|><|start_header_id|>system<|end_header_id|>
		You are a helpful intelligent peer review assistant. You have the ability to take scientific papers and write reviews

		RULES:
		1. After examining the scientific document, you have to provide review of paper's idea, and paper's content.
		2. Paper's idea reivew must consist of idea_only_review_rating(description: str, score: int), idea_only_review_content(str), idea_only_review_confidence (description: str, score: int).
		3. Paper's content reivew must consist of review_rating(description: str, score: int), review_content(str), review_confidence (description: str, score: int).
		4. All scores range from 0 to 10.
		5. DONOT finish `description` or any text prematurely in the middle of sentence.
		6. Include complete sentences and donot chop the text.

		<|eot_id|><|start_header_id|>user<|end_header_id|>
		SCIENTIFIC_PAPER_TITLE:
		```
		PAPER_TITLE_CONTENT
		```

		SCIENTIFIC_PAPER_FULL_TEXT:
		```
		PAPER_FULL_TEXT_CONTENT
		```

		Write a peer review for the above scientific work
		<|eot_id|><|start_header_id|>assistant<|end_header_id|>
		"""
	else:
		# prompt to generate open review style review
		input_text = """
		<|begin_of_text|><|start_header_id|>system<|end_header_id|>
		You are a helpful intelligent peer review assistant. You have the ability to take scientific papers and write reviews

		<|eot_id|><|start_header_id|>user<|end_header_id|>
		SCIENTIFIC_PAPER_TITLE:
		```
		PAPER_TITLE_CONTENT
		```

		SCIENTIFIC_PAPER_FULL_TEXT:
		```
		PAPER_FULL_TEXT_CONTENT
		```

		Write a peer review for the above scientific work
		<|eot_id|><|start_header_id|>assistant<|end_header_id|>
		"""

	# str replace
	prompt = input_text.replace("PAPER_TITLE_CONTENT", dataframe.select("paper_title")[sample_idx].item())
	prompt = input_text.replace("PAPER_FULL_TEXT_CONTENT", dataframe.select("paper_content")[sample_idx].item())

	# return the text
	return prompt
