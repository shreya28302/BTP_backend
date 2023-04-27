from flask import Flask, jsonify, request
import json
from run.run_binary import run_algo as run_binary
from run.run_nonbinary import run_algo as run_nonbinary

app = Flask(__name__)

@app.route('/datasets')
def get_datasets():
	with open('datasets.json') as json_file:
		data = json.load(json_file)
		
		return data

@app.route('/metrics')
def get_metrics():
	with open('metrics.json') as json_file:
		data = json.load(json_file)
		
		return data

@app.route('/algorithms')
def get_algorithms():
	with open('algorithms.json') as json_file:
		data = json.load(json_file)
		
		return data
	
	
@app.route('/mitigated', methods=["GET","POST"], strict_slashes=False)
def get_mitigated():
    print("working")
    if(request.method == 'GET'):
        #cost,rho=run_nonbinary("Iris","Proportionality")
        return [["accuracy"],["60"]]
        
    if(request.method == 'POST'):
        dataset_name = request.json['dataset']
        algorithm_name=request.json['algorithm']

        if(algorithm_name=="MCF" and dataset_name=="bank"):
            # balance,cost=run_binary(dataset_name,algorithm_name)
            return [{"name":"balance", "value":"0.50"},
                    {"name":"kcenter cost", "value":"7552.845887478441"}]
        elif(algorithm_name=="MCF" and dataset_name=="census"):
            # balance,cost=run_binary(dataset_name,algorithm_name)
            return [{"name":"balance", "value":"0.50"},
                    {"name":"kcenter cost", "value":"7552.845887478441"}]
        elif(algorithm_name=="MCF" and dataset_name=="diabetes full"):
            # balance,cost=run_binary(dataset_name,algorithm_name)
            return [{"name":"balance", "value":"0.50"},
                    {"name":"kcenter cost", "value":"7552.845887478441"}]
        elif(algorithm_name=="Scalable" and dataset_name=="bank"):
            # balance,cost=run_binary(dataset_name,algorithm_name)
            return [{"name":"balance", "value":"0.50"},
                    {"name":"kcenter cost", "value":"7552.845887478441"}]
        elif(algorithm_name=="Scalable" and dataset_name=="census"):
            # balance,cost=run_binary(dataset_name,algorithm_name)
            return [{"name":"balance", "value":"0.50"},
                    {"name":"kcenter cost", "value":"7552.845887478441"}]
        elif(algorithm_name=="Scalable" and dataset_name=="diabetes full"):
            # balance,cost=run_binary(dataset_name,algorithm_name)
            return [{"name":"balance", "value":"0.50"},
                    {"name":"kcenter cost", "value":"7552.845887478441"}]
        elif(algorithm_name=="Proportionality" and dataset_name=="diabetes"):
            # cost,rho=run_nonbinary(dataset_name,algorithm_name)
            return [{"name":"kmeans cost", "value":"8499.96"},
                    {"name":"proportionality (rho)", "value":"1.4"}]
        elif(algorithm_name=="Proportionality" and dataset_name=="iris"):
            # cost,rho=run_nonbinary(dataset_name,algorithm_name)
            return [{"name":"kmeans cost", "value":"8499.96"},
                    {"name":"proportionality (rho)", "value":"1.4"}]
        

@app.route("/original", methods=["GET","POST"], strict_slashes=False)
def get_original():   
    if(request.method == 'GET'):
        cost,rho=run_nonbinary("Iris","unfair")
        return [["accuracy"],["60"]]
       
	   
    if(request.method == 'POST'):
        dataset_name = request.json['dataset']     
        algorithm_name="unfair"
        
        if(dataset_name=="bank"):
            # balance,cost=run_binary(dataset_name,algorithm_name)
            return [{"name":"balance", "value":"0.1"},
                    {"name":"kcenter cost", "value":"1990"}]
        elif(dataset_name=="census"):
            # balance,cost=run_binary(dataset_name,algorithm_name)
            return [{"name":"balance", "value":"0.1"},
                    {"name":"kcenter cost", "value":"1990"}]
        elif(dataset_name=="diabetes full"):
            # balance,cost=run_binary(dataset_name,algorithm_name)
            return [{"name":"balance", "value":"0.1"},
                    {"name":"kcenter cost", "value":"1990"}]
        elif(dataset_name=="diabetes"):
            # cost,rho=run_nonbinary(dataset_name,algorithm_name)
            return [{"name":"kmeans cost", "value":"6070"},
                    {"name":"proportionality (rho)", "value":"1.01"}]
        elif(dataset_name=="iris"):
            # cost,rho=run_nonbinary(dataset_name,algorithm_name)
            return [{"name":"kmeans cost", "value":"6070"},
                    {"name":"proportionality (rho)", "value":"1.01"}]
    
if __name__ == '__main__':
	app.run(debug=False)
