const nodemailer = require('nodemailer');
const logger = require('./logger');

const createTransporter = () => {
  return nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_APP_PASSWORD,
    },
  });
};

const sendContactNotification = async (contact) => {
  if (!process.env.EMAIL_USER || !process.env.EMAIL_APP_PASSWORD) {
    logger.warn('Email not configured — skipping notification email.');
    return;
  }

  try {
    const transporter = createTransporter();
    await transporter.sendMail({
      from: `"Portfolio Bot 🤖" <${process.env.EMAIL_USER}>`,
      to: process.env.EMAIL_RECIPIENT,
      subject: `📬 New Contact from ${contact.name} — Portfolio`,
      html: `
        <div style="font-family: 'Arial', sans-serif; max-width: 600px; margin: 0 auto; background: #050816; color: #fff; border-radius: 12px; padding: 32px; border: 1px solid rgba(0,240,255,0.2);">
          <h2 style="color: #00f0ff; margin-top: 0;">📬 New Portfolio Contact</h2>
          <table style="width: 100%; border-collapse: collapse;">
            <tr>
              <td style="padding: 8px; color: #aaa; width: 120px;">Name</td>
              <td style="padding: 8px; color: #fff; font-weight: bold;">${contact.name}</td>
            </tr>
            <tr style="background: rgba(255,255,255,0.05);">
              <td style="padding: 8px; color: #aaa;">Email</td>
              <td style="padding: 8px; color: #00f0ff;">
                <a href="mailto:${contact.email}" style="color: #00f0ff;">${contact.email}</a>
              </td>
            </tr>
            <tr>
              <td style="padding: 8px; color: #aaa;">IP</td>
              <td style="padding: 8px; color: #fff;">${contact.ipAddress}</td>
            </tr>
          </table>
          <div style="margin-top: 24px; padding: 16px; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 3px solid #00f0ff;">
            <p style="color: #aaa; margin: 0 0 8px;">Message:</p>
            <p style="color: #fff; margin: 0; line-height: 1.6;">${contact.message.replace(/\n/g, '<br>')}</p>
          </div>
        </div>
      `,
    });
    logger.info(`✉️  Notification email sent for contact from ${contact.email}`);
  } catch (error) {
    logger.error(`Failed to send notification email: ${error.message}`);
  }
};

const sendAutoReply = async (contact) => {
  if (!process.env.EMAIL_USER || !process.env.EMAIL_APP_PASSWORD) return;

  try {
    const transporter = createTransporter();
    await transporter.sendMail({
      from: `"Sachin Yadav 🚀" <${process.env.EMAIL_USER}>`,
      to: contact.email,
      subject: `Thanks for reaching out, ${contact.name.split(' ')[0]}! — Sachin Yadav`,
      html: `
        <div style="font-family: 'Arial', sans-serif; max-width: 600px; margin: 0 auto; background: #050816; color: #fff; border-radius: 12px; padding: 32px; border: 1px solid rgba(138,43,226,0.3);">
          <h2 style="color: #00f0ff; margin-top: 0;">Hey ${contact.name.split(' ')[0]}! 👋</h2>
          <p style="color: #ccc; line-height: 1.8;">
            Thanks for getting in touch! I've received your message and will get back to you as soon as possible.
          </p>
          <div style="padding: 16px; background: rgba(0,240,255,0.05); border-radius: 8px; margin: 20px 0;">
            <p style="color: #aaa; margin: 0 0 8px; font-size: 12px; text-transform: uppercase;">Your Message</p>
            <p style="color: #fff; margin: 0; font-style: italic;">"${contact.message}"</p>
          </div>
          <p style="color: #555; font-size: 12px; margin: 0;">
            Sachin Yadav — Software Engineer & AI Developer<br>
            Rajkiya Engineering College Kannauj
          </p>
        </div>
      `,
    });
    logger.info(`✉️  Auto-reply sent to ${contact.email}`);
  } catch (error) {
    logger.error(`Failed to send auto-reply: ${error.message}`);
  }
};

module.exports = { sendContactNotification, sendAutoReply };
