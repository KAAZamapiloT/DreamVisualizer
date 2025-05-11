use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use anyhow::Result;
use tracing::{info, error};

// Structure to store feedback data for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackData {
    pub dream_id: String,
    pub dream_text: String,
    pub interpretation: String,
    pub helpful: bool,
    pub comments: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub image_generated: bool,
    pub video_generated: bool,
    pub model_used: Option<String>,
}

// Structure to store model learning data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningData {
    pub feedbacks: Vec<FeedbackData>,
    pub last_updated: DateTime<Utc>,
    pub total_positive: usize,
    pub total_negative: usize,
    pub model_performance: HashMap<String, ModelPerformance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    pub model_name: String,
    pub positive_feedbacks: usize,
    pub negative_feedbacks: usize,
    pub total_uses: usize,
}

pub struct LearningSystem {
    data: Arc<Mutex<LearningData>>,
    data_path: PathBuf,
}

impl LearningSystem {
    pub fn new(data_dir: &str) -> Result<Self> {
        let data_path = Path::new(data_dir).join("learning_data.json");
        let data = if data_path.exists() {
            // Load existing data
            let mut file = File::open(&data_path)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            serde_json::from_str(&contents)?
        } else {
            // Create new data structure
            fs::create_dir_all(data_dir)?;
            LearningData {
                feedbacks: Vec::new(),
                last_updated: Utc::now(),
                total_positive: 0,
                total_negative: 0,
                model_performance: HashMap::new(),
            }
        };

        Ok(Self {
            data: Arc::new(Mutex::new(data)),
            data_path,
        })
    }

    // Add new feedback
    pub fn add_feedback(&self, feedback: FeedbackData) -> Result<()> {
        let mut data = self.data.lock().unwrap();
        
        // Update statistics
        if feedback.helpful {
            data.total_positive += 1;
        } else {
            data.total_negative += 1;
        }
        
        // Update model performance if model is specified
        if let Some(model_name) = &feedback.model_used {
            let model_perf = data.model_performance
                .entry(model_name.clone())
                .or_insert(ModelPerformance {
                    model_name: model_name.clone(),
                    positive_feedbacks: 0,
                    negative_feedbacks: 0,
                    total_uses: 0,
                });
            
            model_perf.total_uses += 1;
            if feedback.helpful {
                model_perf.positive_feedbacks += 1;
            } else {
                model_perf.negative_feedbacks += 1;
            }
        }
        
        // Add feedback to list
        data.feedbacks.push(feedback);
        data.last_updated = Utc::now();
        
        // Save data to disk
        self.save_data(&data)
    }
    
    // Save data to disk
    fn save_data(&self, data: &LearningData) -> Result<()> {
        let json = serde_json::to_string_pretty(data)?;
        let mut file = File::create(&self.data_path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
    
    // Get learning statistics
    pub fn get_statistics(&self) -> Result<LearningStatistics> {
        let data = self.data.lock().unwrap();
        
        let total_feedbacks = data.total_positive + data.total_negative;
        let positive_percentage = if total_feedbacks > 0 {
            (data.total_positive as f64 / total_feedbacks as f64) * 100.0
        } else {
            0.0
        };
        
        // Find best performing model
        let mut best_model = None;
        let mut best_score = 0.0;
        
        for (name, perf) in &data.model_performance {
            if perf.total_uses > 10 { // Minimum threshold for reliability
                let score = perf.positive_feedbacks as f64 / perf.total_uses as f64;
                if score > best_score {
                    best_score = score;
                    best_model = Some((name.clone(), score));
                }
            }
        }
        
        // Get recent feedback comments
        let recent_comments: Vec<&str> = data.feedbacks.iter()
            .filter_map(|f| f.comments.as_deref())
            .rev()
            .take(5)
            .collect();
        
        Ok(LearningStatistics {
            total_feedbacks,
            positive_feedbacks: data.total_positive,
            negative_feedbacks: data.total_negative,
            positive_percentage,
            best_model: best_model.map(|(name, score)| (name, score * 100.0)),
            recent_comments: recent_comments.iter().map(|s| s.to_string()).collect(),
            last_updated: data.last_updated,
        })
    }
    
    // Get learning data for fine-tuning
    pub fn get_training_data(&self) -> Result<Vec<TrainingExample>> {
        let data = self.data.lock().unwrap();
        
        // Convert feedbacks to training examples
        let examples = data.feedbacks.iter()
            .filter(|f| !f.dream_text.is_empty()) // Ensure we have input text
            .map(|f| TrainingExample {
                input: f.dream_text.clone(),
                output: f.interpretation.clone(),
                quality: if f.helpful { 1.0 } else { 0.0 },
            })
            .collect();
            
        Ok(examples)
    }
    
    // Export training data to JSONL for fine-tuning
    pub fn export_training_data(&self, path: &Path) -> Result<usize> {
        let examples = self.get_training_data()?;
        
        if examples.is_empty() {
            return Ok(0);
        }
        
        let mut file = File::create(path)?;
        for example in &examples {
            let json = serde_json::to_string(example)?;
            writeln!(file, "{}", json)?;
        }
        
        Ok(examples.len())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStatistics {
    pub total_feedbacks: usize,
    pub positive_feedbacks: usize,
    pub negative_feedbacks: usize,
    pub positive_percentage: f64,
    pub best_model: Option<(String, f64)>, // (model_name, score_percentage)
    pub recent_comments: Vec<String>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub input: String,
    pub output: String,
    pub quality: f32, // 0.0 to 1.0
}

// GPU-based fine-tuning module
pub mod fine_tuning {
    use super::*;
    use std::process::Command;
    use std::path::Path;
    
    pub struct FineTuner {
        pub model_dir: PathBuf,
        pub training_data_path: PathBuf,
    }
    
    impl FineTuner {
        pub fn new(model_dir: &str, training_data_path: &str) -> Self {
            Self {
                model_dir: PathBuf::from(model_dir),
                training_data_path: PathBuf::from(training_data_path),
            }
        }
        
        // Check if GPU is available
        pub fn is_gpu_available(&self) -> bool {
            let output = Command::new("nvidia-smi")
                .output();
                
            match output {
                Ok(output) => output.status.success(),
                Err(_) => false,
            }
        }
        
        // Start fine-tuning process
        pub fn start_fine_tuning(&self, base_model: &str, output_model: &str) -> Result<()> {
            if !self.is_gpu_available() {
                error!("GPU not available for fine-tuning");
                return Err(anyhow::anyhow!("GPU not available for fine-tuning"));
            }
            
            // Ensure directories exist
            fs::create_dir_all(&self.model_dir)?;
            
            let output_dir = self.model_dir.join(output_model);
            
            info!("Starting fine-tuning process on GPU");
            info!("Base model: {}", base_model);
            info!("Output model: {}", output_dir.display());
            info!("Training data: {}", self.training_data_path.display());
            
            // Run fine-tuning script (this would be implemented separately)
            // For now, we'll just simulate the process
            info!("Fine-tuning process started. This would typically run in the background.");
            
            // In a real implementation, you would spawn a background process to run the fine-tuning
            
            Ok(())
        }
    }
} 