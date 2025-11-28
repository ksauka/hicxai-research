"""
Data Logger for HicXAI Research
Tracks user interactions, application data, and behavior metrics
Saves to private GitHub repository: hicxai-data-private
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import streamlit as st
import requests


class DataLogger:
    """Logs user interactions and saves to private GitHub repository"""
    
    def __init__(self, prolific_id: str, condition: int, session_id: str):
        self.prolific_id = prolific_id
        self.condition = condition
        self.session_id = session_id
        self.session_start = datetime.now().isoformat()
        
        self.interactions: List[Dict] = []
        self.application_data: Dict = {}
        self.behavior_metrics = {
            "total_messages": 0,
            "typed_responses": 0,
            "clicked_responses": 0,
            "help_clicks": 0,
            "explanation_requests": 0,
            "progress_checks": 0,
            "fields_changed": 0
        }
        
    def log_interaction(self, interaction_type: str, content: Dict[str, Any]):
        """Log a single interaction event"""
        self.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            **content
        })
        
        # Update behavior metrics
        if interaction_type == "user_message":
            self.behavior_metrics["total_messages"] += 1
            if content.get("input_method") == "typed":
                self.behavior_metrics["typed_responses"] += 1
            elif content.get("input_method") == "clicked":
                self.behavior_metrics["clicked_responses"] += 1
        elif interaction_type == "help_click":
            self.behavior_metrics["help_clicks"] += 1
        elif interaction_type == "explanation_request":
            self.behavior_metrics["explanation_requests"] += 1
        elif interaction_type == "progress_check":
            self.behavior_metrics["progress_checks"] += 1
    
    def update_application_data(self, field: str, value: Any):
        """Update application field data"""
        if field in self.application_data and self.application_data[field] != value:
            self.behavior_metrics["fields_changed"] += 1
        self.application_data[field] = value
    
    def set_prediction(self, prediction: str, probability: float):
        """Set final prediction result"""
        self.application_data["prediction"] = prediction
        self.application_data["prediction_probability"] = probability
    
    def set_feedback(self, feedback_data: Dict[str, Any]):
        """Set feedback data"""
        self.feedback_data = feedback_data
    
    def build_final_data(self) -> Dict[str, Any]:
        """Build complete data structure for saving"""
        session_end = datetime.now().isoformat()
        start_dt = datetime.fromisoformat(self.session_start)
        end_dt = datetime.fromisoformat(session_end)
        duration = (end_dt - start_dt).total_seconds()
        
        # Get A/B testing info
        try:
            from ab_config import config
            ab_version = config.version
            assistant_name = config.assistant_name
            has_shap = config.show_shap_visualizations
        except:
            ab_version = "unknown"
            assistant_name = "unknown"
            has_shap = False
        
        return {
            "session_id": self.session_id,
            "prolific_id": self.prolific_id,
            "condition": self.condition,
            "ab_version": ab_version,
            "assistant_name": assistant_name,
            "has_shap_visualizations": has_shap,
            "timestamps": {
                "session_start": self.session_start,
                "session_end": session_end,
                "duration_seconds": duration
            },
            "application_data": self.application_data,
            "interactions": self.interactions,
            "behavior_metrics": self.behavior_metrics,
            "feedback": getattr(self, 'feedback_data', None)
        }
    
    def save_to_github(self) -> bool:
        """Save data to private GitHub repository"""
        # Try Streamlit secrets first, then fall back to env variable (for local dev)
        try:
            github_token = st.secrets.get("GITHUB_DATA_TOKEN") or st.secrets.get("GITHUB_TOKEN")
        except:
            github_token = os.getenv('GITHUB_TOKEN')
        
        if not github_token:
            # Fallback to local save
            return self._save_local()
        
        try:
            repo = "ksauka/hicxai-data-private"
            date_str = datetime.now().strftime('%Y-%m-%d')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sessions/{date_str}/{self.prolific_id}_{self.condition}_{timestamp}.json"
            
            data = self.build_final_data()
            content = json.dumps(data, indent=2)
            
            # GitHub API: Create or update file
            url = f"https://api.github.com/repos/{repo}/contents/{filename}"
            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Check if file exists
            response = requests.get(url, headers=headers)
            sha = response.json().get("sha") if response.status_code == 200 else None
            
            # Create/update file
            import base64
            payload = {
                "message": f"Session data: {self.prolific_id} condition {self.condition}",
                "content": base64.b64encode(content.encode()).decode()
            }
            if sha:
                payload["sha"] = sha
            
            response = requests.put(url, headers=headers, json=payload)
            
            if response.status_code in [200, 201]:
                return True
            else:
                # Fallback to local
                return self._save_local()
                
        except Exception as e:
            print(f"GitHub save failed: {e}")
            return self._save_local()
    
    def _save_local(self) -> bool:
        """Fallback: Save to local file"""
        try:
            os.makedirs('data/sessions', exist_ok=True)
            date_str = datetime.now().strftime('%Y-%m-%d')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/sessions/{date_str}_{self.prolific_id}_{self.condition}_{timestamp}.json"
            
            data = self.build_final_data()
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Local save failed: {e}")
            return False


def init_logger() -> Optional[DataLogger]:
    """Initialize data logger from query parameters"""
    if "data_logger" in st.session_state:
        return st.session_state.data_logger
    
    try:
        # Get query params
        try:
            qs = dict(st.query_params)
        except:
            qs = st.experimental_get_query_params()
        
        def _as_str(v):
            return v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else "")
        
        # Extract Prolific ID and condition
        prolific_id = _as_str(qs.get("pid") or qs.get("PROLIFIC_PID", "unknown"))
        condition_str = _as_str(qs.get("cond", "0"))
        condition = int(condition_str) if condition_str.isdigit() else 0
        
        # Generate session ID
        from ab_config import config
        session_id = config.session_id
        
        logger = DataLogger(prolific_id, condition, session_id)
        st.session_state.data_logger = logger
        
        return logger
        
    except Exception as e:
        print(f"Failed to initialize logger: {e}")
        return None
