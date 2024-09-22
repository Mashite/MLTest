using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;
using System.Collections.Generic;
using MLTest;

class Program
{
	static void Main(string[] args)
	{
		var mlContext = new MLContext();

		// Load data from a CSV file
		var data = mlContext.Data.LoadFromTextFile<ProductLengthData>(
			"product_length_data.csv", hasHeader: true, separatorChar: ',');

		// Create a pipeline for clustering using K-Means
		var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Index", "Value" })
						.Append(mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 3));

		// Train the model
		var model = pipeline.Fit(data);

		// Cluster the entire dataset and get the predicted clusters
		var transformedData = model.Transform(data);

		// Extract the predictions
		var predictions = mlContext.Data.CreateEnumerable<ClusterPrediction>(transformedData, reuseRowObject: false).ToList();

		// Combine original data with predictions
		var combinedData = CombineDataWithPredictions(mlContext, data, transformedData);

		// Output the predicted clusters with their respective original indices
		Console.WriteLine("Predicted clusters with corresponding product length indices:");
		foreach (var item in combinedData)
		{
			Console.WriteLine($"Index: {item.Index}, Value: {item.Value}, Predicted Cluster: {item.PredictedCluster}");
		}

		// Find which cluster is most likely to represent 120mm products
		var probableCluster = FindMostProbableCluster(combinedData);
		Console.WriteLine($"\nThe most probable cluster for the correct product length is: {probableCluster}");
	}

	// Combine original data with cluster predictions
	static List<(float Index, float Value, uint PredictedCluster)> CombineDataWithPredictions(
		MLContext mlContext, IDataView originalData, IDataView transformedData)
	{
		var originalDataEnumerable = mlContext.Data.CreateEnumerable<ProductLengthData>(originalData, reuseRowObject: false).ToList();
		var predictions = mlContext.Data.CreateEnumerable<ClusterPrediction>(transformedData, reuseRowObject: false).ToList();

		var combinedData = originalDataEnumerable.Zip(predictions, (original, prediction) =>
			(original.Index, original.Value, prediction.PredictedCluster)).ToList();

		return combinedData;
	}

	// Find the most probable cluster (based on the peak counts in the cluster)
	static uint FindMostProbableCluster(List<(float Index, float Value, uint PredictedCluster)> combinedData)
	{
		// Group by clusters and find the cluster with the highest sum of Value
		var clusterGroups = combinedData
			.GroupBy(data => data.PredictedCluster)
			.Select(group => new
			{
				Cluster = group.Key,
				TotalValue = group.Sum(item => item.Value)  // Sum of counts in the cluster
			})
			.OrderByDescending(group => group.TotalValue)  // Sort by the highest total count
			.FirstOrDefault();

		return clusterGroups?.Cluster ?? 0;
	}
}
