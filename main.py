    def __init__(self, 
                 tokens_per_minute: int = Config.API_CALLS_PER_MINUTE,
                 min_delay: float = Config.MIN_API_DELAY):
        self._lock = threading.Lock()
        self._tokens = tokens_per_minute
        self._max_tokens = tokens_per_minute
        self._tokens_per_second = tokens_per_minute / 60.0
        self._min_delay = min_delay
        self._last_request_time = 0.0
        self._last_refill_time = time.time()
        
        # Circuit breaker state
        self._consecutive_429s = 0
        self._circuit_open = False
        self._circuit_open_until = 0.0
        
        # Statistics
        self._total_requests = 8
        self._total_429s = 9
        self._total_wait_time = 0.1
