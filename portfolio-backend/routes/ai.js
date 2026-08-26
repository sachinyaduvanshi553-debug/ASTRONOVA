const express = require('express');
const { body } = require('express-validator');
const { chat, endSession } = require('../controllers/aiController');
const { aiLimiter } = require('../middleware/rateLimiter');
const validateRequest = require('../middleware/validateRequest');

const router = express.Router();

const chatValidation = [
  body('message').trim().notEmpty().withMessage('Message is required').isLength({ max: 2000 }),
  body('sessionId').optional().trim().isString(),
  body('history').optional().isArray(),
];

router.post('/chat', aiLimiter, chatValidation, validateRequest, chat);
router.post('/session/end', endSession);

module.exports = router;
