import inspect
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from training_data import split_groups,balanced_subset,stage_batches,content_key
from run_experiment import validate_request


class DataProtocolTests(unittest.TestCase):
    def test_holdout_is_disjoint_and_independent_of_training_seed(self):
        rows=[{'id':f'{g}-{c}','prompt_id':str(g),'model':str(c),'text':f'{g} {c}'} for g in range(15) for c in range(18)]
        train,val,ids=split_groups(rows)
        self.assertFalse({r['prompt_id'] for g in train for r in g}&set(ids))
        self.assertEqual(sum(map(len,train))+sum(map(len,val)),len(rows))

    def test_balanced_selection_has_no_replacement_and_ignores_input_order(self):
        rows=[{'model':str(c),'text':f'unique sample {c} {n}'} for c in range(18) for n in range(c+2)]
        a=balanced_subset(rows,list(map(str,range(18))),180,42)
        b=balanced_subset(list(reversed(rows)),list(map(str,range(18))),180,42)
        self.assertEqual(a,b)
        self.assertEqual(len(a),len({r['text'] for r in a}))
        self.assertEqual(len({r['model'] for r in a}),18)

    def test_stage2_triplets_never_cross_prompt_boundaries(self):
        groups=[[{'prompt_id':str(g),'tokens':[1]*n,'text':f'{g}-{n}'} for n in range(1,19)] for g in range(8)]
        batches=stage_batches(groups,42,True)
        for batch in batches:
            for i in range(0,18,3):self.assertEqual(len({r['prompt_id'] for r in batch[i:i+3]}),1)
        public=stage_batches(groups,42,False)
        self.assertEqual({r['text'] for b in public for r in b},{r['text'] for g in groups for r in g})

    def test_content_dedup_catches_format_only_changes(self):
        self.assertEqual(content_key('Hello, world!'),content_key('HELLO — world.'))
        self.assertNotEqual(content_key('Hello world'),content_key('Hello mars'))

    def test_request_requires_pinned_profile_and_bounded_two_stage_fields(self):
        path=Path(__file__).resolve().parents[1]/'experiments/two-stage.example.json'
        request=json.loads(path.read_text())
        validate_request(request)
        for key,value in [('data_profile','/tmp/anything'),('pretrain_max_rows',10000000),('pretrain_epochs',3),('max_minutes',721),('schema_version',1),('pretrain_learning_rate',float('nan'))]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request({**request,key:value})


try:
    import numpy as np
    import sklearn
    from blind_inference import predict_from_features,blind_groups
    NUMPY_AVAILABLE=True
except ImportError:
    NUMPY_AVAILABLE=False


@unittest.skipUnless(NUMPY_AVAILABLE,'Requires numpy/scikit-learn')
class BlindInferenceTests(unittest.TestCase):
    def test_no_gold_group_or_label_argument_exists(self):
        names=set(inspect.signature(predict_from_features).parameters)
        self.assertFalse(names&{'prompt_id','prompt_ids','groups','labels','targets','truth'})

    def test_prediction_is_permutation_equivariant_and_probabilities_are_valid(self):
        texts=['quantum electron energy atomic shell '+x for x in ['orbit','photon','particle']]+['renaissance poetry italian verse '+x for x in ['sonnet','rhyme','rhythm']]
        rng=np.random.default_rng(2);features=rng.normal(size=(6,4)).astype('float32')
        pw=rng.normal(size=(18,4)).astype('float32');cw=rng.normal(size=(18,4)).astype('float32');bias=np.zeros(18,dtype='float32')
        a=predict_from_features(texts,features,pw,bias,cw,bias)
        perm=np.array([5,1,3,0,4,2]);b=predict_from_features([texts[i] for i in perm],features[perm],pw,bias,cw,bias)
        np.testing.assert_allclose(a['probabilities'][perm],b['probabilities'],atol=1e-6)
        np.testing.assert_allclose(a['probabilities'].sum(1),1,atol=1e-6)

    def test_singletons_fall_back_to_plain_prediction(self):
        f=np.ones((1,4),dtype='float32');w=np.ones((18,4),dtype='float32');b=np.zeros(18,dtype='float32')
        p=predict_from_features(['short single passage'],f,w,b,w,b)
        np.testing.assert_array_equal(p['probabilities'],p['plain_probabilities'])
        self.assertEqual(p['centered_rows'],0)


if __name__=='__main__':unittest.main()
