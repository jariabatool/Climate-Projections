from flask import Flask, send_file, request, render_template, jsonify, redirect, flash, url_for, session
import os
from bias_correction import run_bias_correction
import glob

app = Flask(__name__)
app.config['SECRET_KEY'] = 'jaria000'

app.config['PLOTS_FOLDER'] = 'static/plots'
app.config['REPORT_FOLDER'] = 'static/reports'
os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)


def clean_up_old_files(folder_path):
    files = glob.glob(os.path.join(folder_path, '*'))
    for file in files:
        os.remove(file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run_notebook', methods=['POST'])
def run_notebook_endpoint():
    try:
        if request.method == 'POST':
            city = request.form.get('city')
            variable = request.form.get('variable')
            start_time = request.form.get('startTime')
            end_time = request.form.get('endTime')
            

            
            result = run_bias_correction(city, variable, start_time, end_time)
            # errors
            if 'errors' in result:
                return jsonify({'errors': result['errors']})
            
            plots = {
                'temperature_plot': result.get('temperature_plot'),
                'scatter_plot': result.get('scatter_plot'),
                'prediction_plot': result.get('prediction_plot'),
                'predictions_csv': result.get('predictions_csv')
            }
            session['plots'] = plots

        
            return redirect(url_for('results'))
        return render_template("index.html")

    except Exception as e:
        
        flash(f'Error: {str(e)}', 'error')
        print(f"Exception occurred: {str(e)}")  
        return redirect(url_for('index'))


@app.route('/results')
def results():
    plots = session.get('plots', {})
    csv_file = plots.get('predictions_csv', None)
    return render_template('results.html', plots=plots, csv_file=csv_file)

@app.route('/download_csv')
def download_csv():
    csv_file_path = 'static/reports/LR_RF.csv'
    if os.path.exists(csv_file_path):
        return send_file(csv_file_path, as_attachment=True)
    else:
        flash('CSV file not found!', 'error')
        return redirect(url_for('results'))


@app.route('/download_report')
def download_report():
    path_to_report = "static/reports/report.pdf"
    return send_file(path_to_report, as_attachment=True)


if __name__ == '__main__':
    
    clean_up_old_files(app.config['PLOTS_FOLDER'])
    clean_up_old_files(app.config['REPORT_FOLDER'])
    app.run(debug=True)
