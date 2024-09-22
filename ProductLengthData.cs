using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLTest
{
	public class ProductLengthData
	{
		[LoadColumn(0)]
		public float Index { get; set; }

		[LoadColumn(1)]
		public float Value { get; set; }
	}

	public class ClusterPrediction
	{
		[ColumnName("PredictedLabel")]
		public uint PredictedCluster { get; set; }

		[ColumnName("Score")]
		public float[] Distances { get; set; }
	}

}
