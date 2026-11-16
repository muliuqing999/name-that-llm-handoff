"""Validate the exact two-column Parquet output before Kaggle submission."""
def validate_submission(frame, test_ids, labels):
    if list(frame.columns)!=['id','model']:
        raise ValueError('submission must have exactly id, model columns in that order')
    if frame.isna().any().any() or frame.id.duplicated().any():
        raise ValueError('Submission has nulls or repeated ids')
    if len(frame)!=len(test_ids) or set(frame.id)!=set(test_ids):
        raise ValueError('Submission must contain every test id exactly once')
    if not frame.model.isin(labels).all():
        raise ValueError('Submission contains an unknown model label')
    return {'rows':len(frame),'valid':True}
