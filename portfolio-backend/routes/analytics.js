const express = require('express');
const { body } = require('express-validator');
const { trackEvent, upsertSession } = require('../controllers/analyticsController');
const validateRequest = require('../middleware/validateRequest');

const router = express.Router();

const eventValidation = [
  body('sessionId').trim().notEmpty().withMessage('sessionId is required'),
  body('eventType').trim().notEmpty().withMessage('eventType is required'),
  body('page').optional().trim(),
  body('section').optional().trim(),
  body('element').optional().trim(),
  body('metadata').optional().isObject(),
  body('dwellTimeMs').optional().isNumeric(),
];

const sessionValidation = [
  body('sessionId').trim().notEmpty().withMessage('sessionId is required'),
  body('referrer').optional().trim(),
  body('utmSource').optional().trim(),
  body('totalTimeOnSiteMs').optional().isNumeric(),
];

router.post('/event', eventValidation, validateRequest, trackEvent);
router.post('/session', sessionValidation, validateRequest, upsertSession);

module.exports = router;
