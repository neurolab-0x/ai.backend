from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# Global database toggle
# Default to False to ensure no accidental DB interactions
ENABLE_DATABASES = os.getenv('ENABLE_DATABASES', 'false').lower() == 'true'

# Note: InfluxDB and MongoDB configurations were removed as part of the 
# refactor to simplify the system and remove direct DB dependencies.
 