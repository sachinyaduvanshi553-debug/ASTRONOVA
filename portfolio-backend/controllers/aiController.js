const { GoogleGenerativeAI } = require('@google/generative-ai');
const { v4: uuidv4 } = require('uuid');
const AiSession = require('../models/AiSession');
const AiMessage = require('../models/AiMessage');
const logger = require('../utils/logger');

const SYSTEM_PROMPT = `You are Sachin Yadav's personal AI portfolio assistant. Your role is to help recruiters, collaborators, and visitors learn about Sachin.

**About Sachin:**
- Full Stack MERN Developer and AI Enthusiast
- B.Tech Computer Science at Rajkiya Engineering College Kannauj (2021–2025)
- Intern at Developer Street Pvt. Ltd. (Full Stack MERN Developer)
- Hackathon winner: Hack x Vidyouth (1st place), Finalist: Sankalp Innovation Hackathon
- 100 Days DSA Challenge completed, 500+ LeetCode problems solved

**Technical Skills:**
- Languages: Java, Python, JavaScript, C++
- Frontend: React, Next.js, Tailwind CSS
- Backend: Node.js, Express.js, MongoDB, MySQL, REST APIs
- AI/ML: PyTorch, Scikit-Learn, Sentence Transformers, FastAPI

**Projects:**
1. Career Setu AI — AI-powered career guidance platform
2. AstroNova — 3D space exploration visualizer & solar flare forecasting
3. AI Trading Platform — ML-driven stock prediction dashboard
4. Loan Management System — MERN stack financial portal

**Contact:**
- Email: sachinyaduvanshi553@gmail.com
- LinkedIn: https://linkedin.com/in/sachin-yadav-552a35288
- GitHub: https://github.com/sachinyaduvanshi553-debug`;

const chat = async (req, res, next) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  const sendEvent = (data) => {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  const sessionId = req.body.sessionId || uuidv4();
  const userMessage = req.body.message?.trim();
  const history = req.body.history || [];
  const startTime = Date.now();

  if (!userMessage) {
    sendEvent({ type: 'error', message: 'Message is required' });
    return res.end();
  }

  let session;
  let fullResponse = '';
  let chunkCount = 0;

  try {
    session = await AiSession.findOneAndUpdate(
      { sessionId },
      {
        $setOnInsert: {
          sessionId,
          ipAddress: req.ip || 'unknown',
          userAgent: req.headers['user-agent'] || 'unknown',
          status: 'active',
        },
      },
      { upsert: true, new: true }
    );

    await AiMessage.create({
      sessionId,
      role: 'user',
      content: userMessage,
    });

    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({
      model: process.env.GEMINI_MODEL || 'gemini-1.5-flash',
      systemInstruction: SYSTEM_PROMPT,
    });

    const geminiHistory = history.map((h) => ({
      role: h.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: h.content }],
    }));

    const chatSession = model.startChat({ history: geminiHistory });
    const result = await chatSession.sendMessageStream(userMessage);

    for await (const chunk of result.stream) {
      const chunkText = chunk.text();
      if (chunkText) {
        fullResponse += chunkText;
        chunkCount++;
        sendEvent({ type: 'chunk', content: chunkText, sessionId });
      }
    }

    const aggregated = await result.response;
    const usage = aggregated.usageMetadata || {};
    const responseTimeMs = Date.now() - startTime;

    const assistantDoc = await AiMessage.create({
      sessionId,
      role: 'assistant',
      content: fullResponse,
      modelUsed: process.env.GEMINI_MODEL || 'gemini-1.5-flash',
      tokensUsed: (usage.promptTokenCount || 0) + (usage.candidatesTokenCount || 0),
      promptTokens: usage.promptTokenCount || 0,
      completionTokens: usage.candidatesTokenCount || 0,
      responseTimeMs,
      streamChunks: chunkCount,
      wasStreamed: true,
    });

    await AiSession.updateOne(
      { sessionId },
      {
        $inc: {
          totalMessages: 2,
          totalTokensUsed: assistantDoc.tokensUsed,
        },
      }
    );

    sendEvent({
      type: 'done',
      sessionId,
      messageId: assistantDoc._id,
      tokensUsed: assistantDoc.tokensUsed,
      responseTimeMs,
    });

    res.end();
  } catch (error) {
    logger.error(`AI chat error: ${error.message}`);
    sendEvent({ type: 'error', message: 'AI service temporarily unavailable.' });
    res.end();
  }
};

const endSession = async (req, res, next) => {
  try {
    const { sessionId } = req.body;
    const session = await AiSession.findOneAndUpdate(
      { sessionId },
      { status: 'completed', endedAt: new Date() },
      { new: true }
    );
    res.json({ success: true, message: 'Session ended', data: { sessionId } });
  } catch (error) {
    next(error);
  }
};

const getAiSessions = async (req, res, next) => {
  try {
    const { page = 1, limit = 20 } = req.query;
    const [sessions, total] = await Promise.all([
      AiSession.find()
        .sort({ createdAt: -1 })
        .skip((page - 1) * limit)
        .limit(parseInt(limit))
        .lean(),
      AiSession.countDocuments(),
    ]);
    res.json({ success: true, data: sessions, pagination: { total, page: parseInt(page) } });
  } catch (error) {
    next(error);
  }
};

const getSessionMessages = async (req, res, next) => {
  try {
    const messages = await AiMessage.find({ sessionId: req.params.sessionId })
      .sort({ createdAt: 1 })
      .lean();
    res.json({ success: true, data: messages, count: messages.length });
  } catch (error) {
    next(error);
  }
};

module.exports = { chat, endSession, getAiSessions, getSessionMessages };
