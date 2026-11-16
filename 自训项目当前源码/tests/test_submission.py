import importlib.util
import unittest
from scripts.validate_submission import validate_submission

@unittest.skipUnless(importlib.util.find_spec('pandas'),'Requires pandas')
class SubmissionTests(unittest.TestCase):
    def test_rejects_missing_extra_duplicate_null_and_wrong_labels(self):
        import pandas as pd
        good=pd.DataFrame({'id':['a','b'],'model':['x','y']})
        self.assertTrue(validate_submission(good,['a','b'],['x','y'])['valid'])
        cases=[good.iloc[:1],pd.DataFrame({'id':['a','a'],'model':['x','y']}),
               pd.DataFrame({'id':['a','b'],'model':['x','z']}),
               pd.DataFrame({'id':['a','b'],'model':['x',None]}),
               good.assign(probability=0.5),good.rename(columns={'model':'prediction'})]
        for frame in cases:
            with self.assertRaises(ValueError):validate_submission(frame,['a','b'],['x','y'])

if __name__=='__main__':unittest.main()
