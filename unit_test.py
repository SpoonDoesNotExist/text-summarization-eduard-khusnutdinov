import unittest
from unittest.mock import patch, MagicMock
from transformers import pipeline

class TestSummarizationApp(unittest.TestCase):

    @patch("streamlit.text_area")
    @patch("streamlit.button")
    @patch("streamlit.spinner")
    @patch("streamlit.success")
    @patch("streamlit.write")
    @patch("streamlit.error")
    @patch("transformers.pipeline")
    def test_generate_summary(self, mock_pipeline, mock_error, mock_write, mock_success, mock_spinner, mock_button, mock_text_area):

        mock_button.return_value = True
        mock_text_area.return_value = "This is a test input text that needs to be summarized." 

        mock_summarizer = MagicMock()
        mock_summarizer.return_value = [{"summary_text": "This is a test summary."}]
        mock_pipeline.return_value = mock_summarizer

        with patch("streamlit.cache_resource", lambda func: func):
            summarizer = pipeline("summarization")
            summary = summarizer(
                "Streamlit is an open-source Python library that makes it easy to create and share custom web apps for machine learning and data science. By using Streamlit you can quickly build and deploy powerful data applications. For more information about the open-source library, see the Streamlit Library documentation.",
                max_length=130,
                min_length=30,
                do_sample=False
            )[0]["summary_text"]

            self.assertEqual(summary, " Streamlit is an open-source Python library that makes it easy to create and share custom web apps for machine learning and data science . By using Streamlit you can quickly build and deploy powerful data applications .")


    @patch("streamlit.text_area")
    @patch("streamlit.button")
    @patch("streamlit.error")
    def test_empty_input(self, mock_error, mock_button, mock_text_area):

        mock_button.return_value = True
        mock_text_area.return_value = ""

        with patch("streamlit.cache_resource", lambda func: func):
            pipeline("summarization")
            mock_error.assert_called_once_with("Please enter some text to summarize.")

if __name__ == "__main__":
    unittest.main()
