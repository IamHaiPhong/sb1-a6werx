import React, { useState, useEffect } from 'react';
import { Brain, BarChart } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

interface TeamStats {
  wins: number;
  losses: number;
  pointsPerGame: number;
  reboundsPerGame: number;
  assistsPerGame: number;
}

// NBA dataset from 2021-2022 season
const nbaData = [
  [52, 30, 112.1, 44.3, 25.2, 0.634],
  [51, 31, 115.9, 46.0, 28.7, 0.622],
  [56, 26, 111.7, 46.0, 23.8, 0.683],
  [53, 29, 112.1, 44.2, 27.4, 0.646],
  [51, 31, 110.0, 42.7, 23.4, 0.622],
  [44, 38, 112.0, 45.3, 25.2, 0.537],
  [48, 34, 112.1, 43.8, 25.4, 0.585],
  [46, 36, 109.9, 45.3, 25.0, 0.561],
  [44, 38, 108.6, 45.3, 23.7, 0.537],
  [49, 33, 106.6, 45.3, 25.4, 0.598],
  [46, 36, 110.3, 42.0, 25.4, 0.561],
  [44, 38, 109.8, 45.6, 25.2, 0.537],
  [48, 34, 110.0, 44.3, 27.8, 0.585],
  [42, 40, 111.5, 45.0, 24.8, 0.512],
  [36, 46, 108.4, 43.7, 23.7, 0.439],
  [27, 55, 103.7, 43.0, 22.0, 0.329],
  [22, 60, 104.8, 43.5, 21.9, 0.268],
  [25, 57, 106.6, 42.8, 23.4, 0.305],
  [24, 58, 108.0, 42.0, 25.2, 0.293],
  [20, 62, 103.9, 42.8, 22.0, 0.244],
];

function App() {
  const [teamStats, setTeamStats] = useState<TeamStats>({
    wins: 0,
    losses: 0,
    pointsPerGame: 0,
    reboundsPerGame: 0,
    assistsPerGame: 0,
  });
  const [prediction, setPrediction] = useState<number | null>(null);
  const [model, setModel] = useState<tf.Sequential | null>(null);
  const [chartData, setChartData] = useState<any>(null);

  useEffect(() => {
    createAndTrainModel();
  }, []);

  const createAndTrainModel = async () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, inputShape: [5], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

    // Shuffle and split the data
    const shuffledData = tf.util.shuffle(nbaData);
    const splitIndex = Math.floor(shuffledData.length * 0.8);
    const trainData = shuffledData.slice(0, splitIndex);
    const testData = shuffledData.slice(splitIndex);

    const trainX = tf.tensor2d(trainData.map(d => d.slice(0, 5)));
    const trainY = tf.tensor2d(trainData.map(d => [d[5]]));
    const testX = tf.tensor2d(testData.map(d => d.slice(0, 5)));
    const testY = tf.tensor2d(testData.map(d => [d[5]]));

    await model.fit(trainX, trainY, {
      epochs: 200,
      validationData: [testX, testY],
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch}: loss = ${logs?.loss}, val_loss = ${logs?.val_loss}`);
        }
      }
    });

    setModel(model);

    // Evaluate the model
    const evalResult = model.evaluate(testX, testY) as tf.Scalar[];
    console.log(`Evaluation result: ${evalResult[0].dataSync()[0]}`);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    const numValue = parseFloat(value);
    setTeamStats((prevStats) => ({
      ...prevStats,
      [name]: isNaN(numValue) ? 0 : numValue,
    }));
  };

  const normalizeInput = (input: number[]) => {
    const maxValues = [82, 82, 120, 50, 30]; // Adjusted max values based on NBA data
    return input.map((val, index) => val / maxValues[index]);
  };

  const predictWinPercentage = () => {
    if (!model) {
      alert("Model is not ready yet. Please try again in a moment.");
      return;
    }

    const totalGames = teamStats.wins + teamStats.losses;
    if (totalGames === 0) {
      alert("Please enter at least one win or loss.");
      return;
    }

    const input = normalizeInput([
      teamStats.wins,
      teamStats.losses,
      teamStats.pointsPerGame,
      teamStats.reboundsPerGame,
      teamStats.assistsPerGame
    ]);

    const inputTensor = tf.tensor2d([input]);
    const predictedTensor = model.predict(inputTensor) as tf.Tensor;
    const predictedValue = predictedTensor.dataSync()[0];
    const winPercentage = Math.round(predictedValue * 100);
    setPrediction(Math.max(0, Math.min(100, winPercentage))); // Ensure the prediction is between 0 and 100

    // Generate data for visualization
    const visualizationData = generateVisualizationData();
    setChartData(visualizationData);
  };

  const generateVisualizationData = () => {
    if (!model) return null;

    const baseStats = { ...teamStats };
    const labels = ['Points', 'Rebounds', 'Assists'];
    const datasets = labels.map((label, index) => {
      const data = [];
      for (let i = -5; i <= 5; i++) {
        const tempStats = { ...baseStats };
        const key = `${label.toLowerCase()}PerGame` as keyof TeamStats;
        tempStats[key] = Math.max(0, baseStats[key] + i);
        const input = normalizeInput([
          tempStats.wins,
          tempStats.losses,
          tempStats.pointsPerGame,
          tempStats.reboundsPerGame,
          tempStats.assistsPerGame
        ]);
        const inputTensor = tf.tensor2d([input]);
        const predictedTensor = model.predict(inputTensor) as tf.Tensor;
        const predictedValue = predictedTensor.dataSync()[0];
        const winPercentage = Math.round(predictedValue * 100);
        data.push(Math.max(0, Math.min(100, winPercentage)));
      }
      return {
        label,
        data,
        borderColor: ['#3b82f6', '#10b981', '#f59e0b'][index],
        backgroundColor: ['rgba(59, 130, 246, 0.5)', 'rgba(16, 185, 129, 0.5)', 'rgba(245, 158, 11, 0.5)'][index],
      };
    });

    return {
      labels: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].map(String),
      datasets,
    };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center p-4">
      <div className="bg-white p-8 rounded-lg shadow-xl w-full max-w-4xl">
        <h1 className="text-3xl font-bold mb-6 text-center flex items-center justify-center text-gray-800">
          <Brain className="mr-2 text-blue-500" /> NBA Win % Predictor
        </h1>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-4">
            {Object.entries(teamStats).map(([key, value]) => (
              <div key={key} className="flex flex-col">
                <label htmlFor={key} className="text-sm font-medium text-gray-700 mb-1">
                  {key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1')}
                </label>
                <input
                  type="number"
                  id={key}
                  name={key}
                  value={value || ''}
                  onChange={handleInputChange}
                  className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  min="0"
                  step={key === 'wins' || key === 'losses' ? "1" : "0.1"}
                />
              </div>
            ))}
            <button
              onClick={predictWinPercentage}
              className="w-full bg-blue-500 text-white p-3 rounded-md hover:bg-blue-600 transition-colors duration-300 font-semibold"
            >
              Predict Win Percentage
            </button>
          </div>
          <div>
            {prediction !== null && (
              <div className="mb-6 text-center">
                <h2 className="text-xl font-semibold mb-2 flex items-center justify-center text-gray-800">
                  <BarChart className="mr-2 text-blue-500" /> Predicted Win %
                </h2>
                <p className="text-4xl font-bold text-blue-600">{prediction}%</p>
              </div>
            )}
            {chartData && (
              <div className="mt-4">
                <h3 className="text-lg font-semibold mb-2 text-center text-gray-800">AI Prediction Visualization</h3>
                <Line
                  data={chartData}
                  options={{
                    responsive: true,
                    plugins: {
                      legend: {
                        position: 'top' as const,
                      },
                      title: {
                        display: true,
                        text: 'Win % Change by Stat Adjustment',
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'Stat Adjustment',
                        },
                      },
                      y: {
                        title: {
                          display: true,
                          text: 'Predicted Win %',
                        },
                        min: 0,
                        max: 100,
                      },
                    },
                  }}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;