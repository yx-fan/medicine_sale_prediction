def static_forecast(model, exog, log_y, train_end, end_date_filter):
    predictions = []
    for t in range(train_end, len(log_y)):
        if log_y.index[t] > end_date_filter:
            break
        exog_current = exog.iloc[t: t + 1].fillna(0)
        next_pred_log = model.predict(n_periods=1, exogenous=exog_current).item()
        model.update([log_y.iloc[t]], exogenous=exog_current)
        predictions.append(next_pred_log)
    return predictions
