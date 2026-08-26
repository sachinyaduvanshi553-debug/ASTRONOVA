const rateLimit = require('express-rate-limit');
const logger = require('../utils/logger');

const generalLimiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000,
  max: parseInt(process.env.RATE_LIMIT_MAX) || 100,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    success: false,
    error: 'Too many requests. Please try again after 15 minutes.',
  },
  handler: (req, res, next, options) => {
    logger.warn(`Rate limit exceeded: IP=${req.ip}, Path=${req.path}`);
    res.status(429).json(options.message);
  },
});

const contactLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  max: 5,
  message: {
    success: false,
    error: 'Too many messages sent. Please wait an hour before sending another message.',
  },
});

const aiLimiter = rateLimit({
  windowMs: 10 * 60 * 1000,
  max: 30,
  message: {
    success: false,
    error: 'AI chat rate limit reached. Please wait a moment before sending more messages.',
  },
});

module.exports = { generalLimiter, contactLimiter, aiLimiter };
