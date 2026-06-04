import io
import unittest
from contextlib import redirect_stdout

import pandas as pd
from sklearn.dummy import DummyRegressor
import stratum as st
from stratum.optimizer._explain import explain_linear_plan
from stratum.optimizer._optimize import optimize
from stratum.optimizer.ir._ops import Op


class TestExplainLinearPlan(unittest.TestCase):

    def test_explain_linear_plan_no_split(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        data = st.var("data", df)
        X = data[["x"]].skb.mark_as_X()
        X = X + 33

        # Let the optimizer construct the graph and linearize it
        ops, split_pos, _ = optimize(X)
        self.assertIsNone(split_pos)

        # Append an external op to the inputs of the first op to test the ?[Op] formatting path
        external_op = Op(name="external_op")
        ops[0].add_input(external_op)

        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            explain_linear_plan("pipeline_no_split", ops, split_pos=None)

        output_str = captured_output.getvalue()
        self.assertIn("=== Plan: pipeline_no_split ===", output_str)
        # Should show that external input is not in the linearized plan list
        self.assertIn("?[Op(external_op)]", output_str)
        print(output_str)

    def test_explain_linear_plan_split_none_empty(self):
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            explain_linear_plan("test_empty", [], split_pos=None)

        output_str = captured_output.getvalue()
        self.assertIn("=== Plan: test_empty ===", output_str)
        print(output_str)

    def test_explain_linear_plan_with_split(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        data = st.var("data", df)
        x = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        x = x + 33
        pred = x.skb.apply(DummyRegressor(), y=y)

        # Let the optimizer construct the graph and linearize it
        ops, split_pos, _ = optimize(pred)
        self.assertIsNotNone(split_pos)

        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            explain_linear_plan("pipeline_with_split", ops, split_pos=split_pos)

        output_str = captured_output.getvalue()
        self.assertIn("=== Plan: pipeline_with_split ===", output_str)
        self.assertIn("CV Loop:", output_str)
        self.assertIn("Fit Phase:", output_str)
        self.assertIn("Transform / Predict:", output_str)
        self.assertIn("Total:", output_str)
        print(output_str)

    def test_explain_linear_plan_integration(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        data = st.var("data", df)
        X = data[["x"]].skb.mark_as_X()

        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            with st.config(explain_linear_plan=True):
                optimize(X)

        output_str = captured_output.getvalue()
        self.assertIn("=== Plan: explain_linear_plan ===", output_str)
        print(output_str)


if __name__ == "__main__":
    unittest.main()
