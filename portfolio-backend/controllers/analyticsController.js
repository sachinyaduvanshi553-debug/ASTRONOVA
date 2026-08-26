const AnalyticsEvent = require('../models/AnalyticsEvent');
const VisitorSession = require('../models/VisitorSession');
const ProjectInteraction = require('../models/ProjectInteraction');
const ResumeDownload = require('../models/ResumeDownload');
const Contact = require('../models/Contact');
const AiSession = require('../models/AiSession');
const logger = require('../utils/logger');

const trackEvent = async (req, res, next) => {
  try {
    const {
      sessionId,
      eventType,
      page = '/',
      section = null,
      element = null,
      metadata = {},
      dwellTimeMs = null,
    } = req.body;

    const ipAddress = req.ip || req.headers['x-forwarded-for'] || 'unknown';
    const userAgent = req.headers['user-agent'] || 'unknown';

    const event = await AnalyticsEvent.create({
      sessionId,
      eventType,
      page,
      section,
      element,
      metadata,
      dwellTimeMs,
      ipAddress,
      userAgent,
    });

    const updatePromises = [];

    if (eventType === 'PROJECT_CLICK' && metadata.projectTitle) {
      updatePromises.push(
        ProjectInteraction.create({
          sessionId,
          projectTitle: metadata.projectTitle,
          interactionType: metadata.interactionType || 'card_view',
          dwellTimeMs,
          ipAddress,
        })
      );
      updatePromises.push(
        VisitorSession.updateOne(
          { sessionId },
          { $addToSet: { projectsInteracted: metadata.projectTitle }, lastSeen: new Date() }
        )
      );
    }

    if (eventType === 'RESUME_DOWNLOAD') {
      updatePromises.push(
        ResumeDownload.create({
          sessionId,
          ipAddress,
          userAgent,
          referrer: metadata.referrer || req.headers.referer || 'direct',
          triggeredFrom: metadata.triggeredFrom || 'unknown',
        })
      );
      updatePromises.push(
        VisitorSession.updateOne(
          { sessionId },
          { didDownloadResume: true, lastSeen: new Date() }
        )
      );
    }

    if (eventType === 'SECTION_VIEW' && section) {
      updatePromises.push(
        VisitorSession.updateOne(
          { sessionId },
          { $addToSet: { sectionsViewed: section }, lastSeen: new Date() }
        )
      );
    }

    Promise.all(updatePromises).catch((err) =>
      logger.error(`Analytics side-effect error: ${err.message}`)
    );

    res.status(201).json({ success: true, data: { id: event._id } });
  } catch (error) {
    next(error);
  }
};

const upsertSession = async (req, res, next) => {
  try {
    const { sessionId, referrer, utmSource, totalTimeOnSiteMs } = req.body;
    const ipAddress = req.ip || req.headers['x-forwarded-for'] || 'unknown';
    const userAgent = req.headers['user-agent'] || 'unknown';

    const device = /mobile/i.test(userAgent)
      ? 'mobile'
      : /tablet|ipad/i.test(userAgent)
      ? 'tablet'
      : 'desktop';

    const session = await VisitorSession.findOneAndUpdate(
      { sessionId },
      {
        $setOnInsert: {
          sessionId,
          ipAddress,
          userAgent,
          device,
          referrer: referrer || 'direct',
          utmSource: utmSource || null,
          firstSeen: new Date(),
        },
        $set: {
          lastSeen: new Date(),
          ...(totalTimeOnSiteMs && { totalTimeOnSiteMs }),
        },
        $inc: { pagesViewed: 1 },
      },
      { upsert: true, new: true }
    );

    res.json({ success: true, data: { sessionId: session.sessionId } });
  } catch (error) {
    next(error);
  }
};

const getAnalyticsSummary = async (req, res, next) => {
  try {
    const { days = 30 } = req.query;
    const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000);

    const [
      totalVisitors,
      recentVisitors,
      eventsByType,
      topProjects,
      totalResumes,
      deviceBreakdown,
    ] = await Promise.all([
      VisitorSession.countDocuments(),
      VisitorSession.countDocuments({ firstSeen: { $gte: since } }),
      AnalyticsEvent.aggregate([
        { $match: { createdAt: { $gte: since } } },
        { $group: { _id: '$eventType', count: { $sum: 1 } } },
        { $sort: { count: -1 } },
      ]),
      ProjectInteraction.aggregate([
        { $match: { createdAt: { $gte: since } } },
        { $group: { _id: '$projectTitle', clicks: { $sum: 1 } } },
        { $sort: { clicks: -1 } },
      ]),
      ResumeDownload.countDocuments({ createdAt: { $gte: since } }),
      VisitorSession.aggregate([
        { $group: { _id: '$device', count: { $sum: 1 } } },
      ]),
    ]);

    res.json({
      success: true,
      data: {
        period: `Last ${days} days`,
        visitors: { total: totalVisitors, recent: recentVisitors },
        eventsByType,
        topProjects,
        resumeDownloads: totalResumes,
        deviceBreakdown,
      },
    });
  } catch (error) {
    next(error);
  }
};

const getDashboardStats = async (req, res, next) => {
  try {
    const [
      totalContacts,
      unreadContacts,
      totalAiSessions,
      totalVisitors,
      totalResumes,
    ] = await Promise.all([
      Contact.countDocuments(),
      Contact.countDocuments({ status: 'unread' }),
      AiSession.countDocuments(),
      VisitorSession.countDocuments(),
      ResumeDownload.countDocuments(),
    ]);

    res.json({
      success: true,
      data: {
        contacts: { total: totalContacts, unread: unreadContacts },
        ai: { totalSessions: totalAiSessions },
        visitors: { total: totalVisitors },
        resume: { downloads: totalResumes },
        generatedAt: new Date(),
      },
    });
  } catch (error) {
    next(error);
  }
};

module.exports = { trackEvent, upsertSession, getAnalyticsSummary, getDashboardStats };
