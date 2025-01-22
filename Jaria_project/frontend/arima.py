import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
matplotlib.use('Agg')
from statsmodels.tsa.arima.model import ARIMA



class BcsdTemperature:
    def __init__(self, return_anoms=False):
        self.return_anoms = return_anoms


def run_bias_correction(city, variable, start_time, end_time):
    """Run bias correction on provided datasets and return results."""
    start_time1 = start_time
    end_time1=end_time
    start_time1 = pd.to_datetime(start_time1)
    end_time1 = pd.to_datetime(end_time1)
    start_year1 = str(start_time1.year)
    end_year1 = str(end_time1.year)

    # Error messages
    error_messages = {'time_range': None, 'data': None}
    results = {
        'temperature_plot': None,
        'scatter_plot': None,
        'prediction_plot': None,
        'descriptions': {
            'outliers': None,
            'cleaned_data': None,
            'model_scores': {},
            'predictions': None
        }
    }

    # Convert dates to datetime
    start_time = pd.to_datetime('01/01/2008')
    end_time = pd.to_datetime('01/01/2015')

    start_year = str(start_time.year)
    end_year = str(end_time.year)

    # Example check for time range
    if end_time <= start_time:
        error_messages['time_range'] = 'End time must be greater than start time.'
        return {'errors': error_messages}
    try:
        train_col = ''
        target_col = ''
        # Load datasets
        file1_path = f'dataset/{city}{variable}Final.csv'
        file2_path = f'dataset/{city}{variable}.csv'

        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)

        df1['time'] = pd.to_datetime(df1['time'])
        df2['Time'] = pd.to_datetime(df2['Time'])

        # Set indices
        df1 = df1.set_index(['time'])
        df2 = df2.set_index(['Time'])

        if df1.empty or df2.empty:
            error_messages['time_range'] = 'No data available for the selected time range.'
            return {'errors': error_messages}

        if variable == 'Tmax':
            train_col = 'tasmax'
            target_col = 'max'
        elif variable == 'Tmin':
            train_col = 'tmin'
            target_col = 'min'
        elif variable == 'prep':
            train_col = 'rain'
            target_col = 'prep'

        # Prepare data
        train_df = list(df2[train_col])
        target_df = list(df1[target_col])

        train_data1 = {'Time': df2.index, train_col: train_df}
        train = pd.DataFrame(train_data1)

        target_data = {'time': df1.index, target_col: target_df}
        target = pd.DataFrame(target_data)

        train_data1 = pd.read_csv(file2_path)
        z_scores = (train_data1[train_col] - train_data1[train_col].mean()) / train_data1[train_col].std()
        threshold = 3
        outliers = train_data1[np.abs(z_scores) > threshold]
        train_data = train_data1[np.abs(z_scores) <= threshold]

        results['descriptions']['outliers'] = outliers.to_dict()
        results['descriptions']['cleaned_data'] = train_data.to_dict()

        # Plot temperature data
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6), sharex=True)
        train_data, target_data = train, target
        train_data[train_col].plot(ax=ax, label='training')
        target_data[target_col].plot(ax=ax, label='targets')
        ax.legend()
        ax.set_ylabel('Temperature [C]')
        temperature_plot_path = 'static/plots/temperature_plot.png'
        plt.savefig(temperature_plot_path)

        plt.close()

        # Data slicing and extraction
        train_data = train_data.set_index(['Time'])
        target_data = target_data.set_index(['time'])

        X = train_data[[train_col]][str(start_year):str(end_year)].values
        y = target_data[[target_col]][str(start_year):str(end_year)].values.ravel()

        min_samples = min(X.shape[0], y.shape[0])
        X = X[:min_samples]
        y = y[:min_samples]
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        if min_samples < 3:
            error_messages['data'] = "Insufficient data for model training."
            return {'errors': error_messages}

        xqt = QuantileTransformer(n_quantiles=1000, copy=True, subsample=1000).fit(X_train)
        Xq_train = xqt.transform(X_train)
        Xq_test = xqt.transform(X_test)

        yqt = QuantileTransformer(n_quantiles=1000, copy=True, subsample=1000).fit(y_train.reshape(-1, 1))
        yyq_train = yqt.transform(y_train.reshape(-1, 1)).ravel()  # Ensure yq_train is 1D after transformation
        yq_test = yqt.transform(y_test.reshape(-1, 1)).ravel()  # Ensure yq_test

        # Plot train-test scatter
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plt.scatter(X_train, y_train, c='black', s=5, label='Train')
        plt.scatter(X_test, y_test, c='grey', s=5, label='Test')
        plt.legend()
        scatter_plot_path = 'static/plots/train_test_scatter.png'
        plt.savefig(scatter_plot_path)
        plt.close()

        # Train models
        models = {
            'GARD: LinearRegression': LinearRegression(),
            'Sklearn: RandomForestRegressor': RandomForestRegressor(random_state=0),
            'ARIMA': None 
        }

        for model_name, model in models.items():
            if model_name == 'ARIMA':
                # ARIMA model needs a little adjustment for univariate time series prediction
                # Fit the ARIMA model
                try:
                    arima_model = ARIMA(y_train, order=(5, 1, 0))  # Adjust order (p, d, q) as needed
                    arima_model_fit = arima_model.fit()
                    # Store ARIMA model score for comparison
                    results['descriptions']['model_scores']['ARIMA'] = arima_model_fit.aic
                except Exception as e:
                    results['descriptions']['model_scores']['ARIMA'] = str(e)
            else:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                results['descriptions']['model_scores'][model_name] = score

        # Make predictions
        
        train_slice = slice(str(start_time), str(end_time))
        predict_slice = slice(str(end_time), '2099-12-31')

        X_train = train_data[[train_col]][train_slice]
        y_train = target_data[[target_col]][train_slice]
        X_predict = train_data[[train_col]][predict_slice]
        min_samples = min(X_train.shape[0], y_train.shape[0])
        X_train = X_train[:min_samples]
        y_train = y_train[:min_samples]
        y_train = y_train.values.ravel()


        if 'ARIMA' in models:
            arima_model = ARIMA(y_train, order=(5, 1, 0))  # Adjust order (p, d, q) as needed
            arima_model_fit = arima_model.fit()
            arima_predictions = arima_model_fit.predict(start=len(y_train), end=len(y_train)+len(X_predict)-1, typ='levels')

        for key, model in models.items():
            if key != 'ARIMA':
                model.fit(X_train, y_train)

        predict_df = pd.DataFrame(index=X_predict.index)
        for name, model in models.items():
            if name == 'ARIMA':
                predict_df[name] = arima_predictions 
            else :   
                predict_df[name] = model.predict(X_predict)

        results['descriptions']['predictions'] = predict_df.to_dict()


        # Plot predictions
        fig, ax = plt.subplots(figsize=(15, 15))
        time_slice = slice(start_year1, end_year1)
        if 'ARIMA' in predict_df.columns:
            predict_df['ARIMA'][time_slice].plot(ax=ax, label='ARIMA Predictions', lw=0.75)

        for model_name in models:
            if model_name != 'ARIMA':
                predict_df[model_name][time_slice].plot(ax=ax, label=f'{model_name} Predictions', lw=0.75)    
        # Plot X_predict[train_col]
        if not X_predict[train_col][time_slice].empty:
            X_predict[train_col][time_slice].plot(label='original', c='grey', ax=ax, alpha=0.75, legend=True)
        else:
            print("X_predict[train_col][time_slice] is empty")

        # Plot predict_df
        if not predict_df[time_slice].empty:
            predict_df[time_slice].plot(ax=ax, lw=0.75)
        else:
            print("predict_df[time_slice] is empty")

        # Plot target data
        if not target_data[target_col][time_slice].empty:
            target_data[target_col][time_slice].plot(ax=ax, label='target', c='k', lw=1, alpha=0.75, legend=True,
                                                     zorder=10)
        else:
            print("target_data[target_col][time_slice] is empty")

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylabel('Temperature [C]')
        # X_predict[train_col][time_slice].plot(label='original', c='grey', ax=ax, alpha=0.75, legend=True)
        # predict_df[time_slice].plot(ax=ax, lw=0.75)
        # target_data[target_col][time_slice].plot(ax=ax, label='target', c='k', lw=1, alpha=0.75, legend=True, zorder=10)
        prediction_plot_path = 'static/plots/prediction_plot.png'
        plt.savefig(prediction_plot_path)
        plt.close()

        predict_df.to_csv('hushe_RF_test_saidusharif.csv')
        
        # Update results with file paths and errors
        results.update({
            'temperature_plot': temperature_plot_path,
            'scatter_plot': scatter_plot_path,
            'prediction_plot': prediction_plot_path,
            'errors': error_messages
        })
        filename = "static/reports/report.pdf"
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()

        # Elements to include in the PDF
        elements = []

        # Title
        elements.append(Paragraph("Climate Projection Report", styles['Title']))
        elements.append(Spacer(1, 12))

        # Description of scatter plot
        elements.append(Paragraph("Train-Test Scatter Plot", styles['Heading2']))
        elements.append(Paragraph(
            "This scatter plot visualizes the relationship between the training data and the target data, distinguishing between training samples and test samples.",
            styles['BodyText']))
        elements.append(Image(scatter_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))

        # Description of temperature plot
        elements.append(Paragraph("Temperature Plot", styles['Heading2']))
        elements.append(Paragraph(
            "This plot compares the original training temperature data ('tasmax') and the target data ('max').",
            styles['BodyText']))
        elements.append(Image(temperature_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))

        # Description of prediction plot
        elements.append(Paragraph("Prediction Plot", styles['Heading2']))
        elements.append(Paragraph(
            "This plot shows the predicted temperature values by different models alongside the original climate data ('tasmax') and the target data ('max').",
            styles['BodyText']))
        elements.append(Image(prediction_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))

        # Outliers detected
        elements.append(Paragraph("Detected Outliers", styles['Heading2']))
        elements.append(Paragraph(str(results['descriptions']['outliers']), styles['BodyText']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Station Selected: ", styles['Heading2']))
        elements.append(Paragraph(city, styles['BodyText']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Variable Selected: ", styles['Heading2']))
        elements.append(Paragraph(variable, styles['BodyText']))
        elements.append(Spacer(1, 12))

        # Model Scores
        elements.append(Paragraph("Model Scores", styles['Heading2']))
        for model_name, score in results['descriptions']['model_scores'].items():
            elements.append(Paragraph(f"{model_name}: {score:.2f}", styles['BodyText']))
        elements.append(Spacer(1, 12))

        try:
            doc.build(elements)
        except Exception as e:
            print(f"An error occurred while generating the PDF: {e}")
        return results

    except Exception as e:
        # Handle general errors
        print("errors:", str(e))
        return {'errors': {'general': str(e)}}
