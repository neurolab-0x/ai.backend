import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PROCESSING_CONFIG = {
    'sample_rate': 0.5,
    'smoothing_window': 5,
    'max_timeline_points': 100,
    # Streaming buffer settings
    'max_buffer_size': 5000,  # Maximum samples to retain in buffer
    'min_window_size': 50,    # Minimum window size for adaptive processing
    'max_window_size': 500,   # Maximum window size for adaptive processing
    'buffer_overlap_percent': 25  # Overlap percentage between consecutive windows
}

THRESHOLDS = {
    'stress_threshold': 0.25,
    'confidence_threshold': 70,
    'severe_stress_threshold': 0.4,
    'relaxation_threshold': 0.6,
    'max_recommendations': 3
}

REAL_TIME_CONFIG = {
    'enable_caching': True,         # Enable model caching
    'enable_streaming': True,       # Enable streaming buffer
    'enable_adaptive_window': True, # Enable adaptive window sizing
    'latency_threshold_ms': 100     # Target latency threshold in milliseconds
}

SECURITY_CONFIG = {
    # ============================================
    # AUTHENTICATION CONTROL
    # ============================================
    # Authentication is DISABLED by default for development
    # 
    # To ENABLE authentication:
    # 1. Set REQUIRE_AUTH=true environment variable, OR
    # 2. Change the default value below to True
    # 
    # For production: Always set REQUIRE_AUTH=true
    # ============================================
    
    # Encryption settings
    'enable_encryption': True,              # Enable data encryption
    'encryption_for_storage': True,         # Encrypt data before storing
    'encryption_for_transit': True,         # Always encrypt responses
    
    # Authentication settings
    'require_authentication': os.getenv('REQUIRE_AUTH', 'false').lower() == 'true',  # Can be overridden with REQUIRE_AUTH env var
    'token_expiry_hours': 24,               # JWT token expiry time in hours
    'refresh_token_expiry_days': 30,        # Refresh token expiry in days
    
    # Authorization settings
    'required_role_realtime': 'user',       # Role required for real-time processing
    'required_role_admin': 'admin',         # Role required for administrative functions
    
    # Rate limiting
    'enable_rate_limiting': True,           # Enable rate limiting
    'rate_limit_requests': 60,              # Maximum requests per window
    'rate_limit_window_seconds': 60,        # Window size in seconds
    
    # Data validation
    'max_eeg_amplitude': 1000,              # Maximum allowed amplitude in microvolts
    'max_eeg_channels': 64,                 # Maximum allowed EEG channels
    'max_eeg_samples': 10000                # Maximum samples per request
}

# Logging configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_security_events': True,
    'security_log_path': './logs/security.log',
    'application_log_path': './logs/application.log'
}

MODEL_VERSION = "v3.1.0"
MODEL_NAME = "NeuroLab Axon Prime"
 
LLM_CONFIG = {
    'provider': 'groq',
    'api_key': os.getenv('GROQ_API_KEY'),
    'model': 'llama-3.1-8b-instant',
    'temperature': 0.7,
    'max_tokens': 1024,
    'timeout': 30
}

OPENROUTER_CONFIG = {
    'base_url': os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
    'api_key': os.getenv('OPENROUTER_API_KEY'),
    'model': os.getenv('OPENROUTER_MODEL', 'openai/gpt-4o-mini'),
    'temperature': float(os.getenv('OPENROUTER_TEMPERATURE', '0.5')),
    'max_tokens': int(os.getenv('OPENROUTER_MAX_TOKENS', '700')),
    'timeout': float(os.getenv('OPENROUTER_TIMEOUT', '60')),
    'app_name': os.getenv('OPENROUTER_APP_NAME', 'NeuroLab AI'),
    'site_url': os.getenv('OPENROUTER_SITE_URL', ''),
}
