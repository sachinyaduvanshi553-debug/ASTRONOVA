const express = require('express');
const adminAuth = require('../middleware/adminAuth');
const { getContacts, updateContactStatus } = require('../controllers/contactController');
const { getAiSessions, getSessionMessages } = require('../controllers/aiController');
const { getAnalyticsSummary, getDashboardStats } = require('../controllers/analyticsController');

const router = express.Router();

router.use(adminAuth);

router.get('/stats', getDashboardStats);
router.get('/contacts', getContacts);
router.patch('/contacts/:id', updateContactStatus);
router.get('/ai-sessions', getAiSessions);
router.get('/ai-sessions/:sessionId/messages', getSessionMessages);
router.get('/analytics', getAnalyticsSummary);

module.exports = router;
