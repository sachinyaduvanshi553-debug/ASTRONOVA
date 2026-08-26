const Contact = require('../models/Contact');
const { sendContactNotification, sendAutoReply } = require('../utils/emailSender');
const logger = require('../utils/logger');

const submitContact = async (req, res, next) => {
  try {
    const { name, email, message } = req.body;

    const contact = await Contact.create({
      name: name.trim(),
      email: email.trim().toLowerCase(),
      message: message.trim(),
      ipAddress: req.ip || req.headers['x-forwarded-for'] || 'unknown',
      userAgent: req.headers['user-agent'] || 'unknown',
    });

    logger.info(`📩 New contact from ${contact.email} [ID: ${contact._id}]`);

    Promise.all([
      sendContactNotification(contact),
      sendAutoReply(contact),
    ]).catch((err) => logger.error(`Email background task failed: ${err.message}`));

    return res.status(201).json({
      success: true,
      message: 'Message received! I will get back to you within 24–48 hours.',
      data: {
        id: contact._id,
        createdAt: contact.createdAt,
      },
    });
  } catch (error) {
    next(error);
  }
};

const getContacts = async (req, res, next) => {
  try {
    const { status, page = 1, limit = 20 } = req.query;
    const filter = status ? { status } : {};

    const [contacts, total] = await Promise.all([
      Contact.find(filter)
        .sort({ createdAt: -1 })
        .skip((page - 1) * limit)
        .limit(parseInt(limit))
        .lean(),
      Contact.countDocuments(filter),
    ]);

    res.json({
      success: true,
      data: contacts,
      pagination: {
        total,
        page: parseInt(page),
        pages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    next(error);
  }
};

const updateContactStatus = async (req, res, next) => {
  try {
    const { status, replyContent } = req.body;
    const update = { status };
    if (status === 'replied' && replyContent) {
      update.replyContent = replyContent;
      update.repliedAt = new Date();
    }

    const contact = await Contact.findByIdAndUpdate(
      req.params.id,
      update,
      { new: true, runValidators: true }
    );

    if (!contact) {
      return res.status(404).json({ success: false, error: 'Contact not found' });
    }

    res.json({ success: true, data: contact });
  } catch (error) {
    next(error);
  }
};

module.exports = { submitContact, getContacts, updateContactStatus };
