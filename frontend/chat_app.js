// AI Agent Chat 应用程序
// 处理流式响应、工具调用和 Human-in-the-loop

//全局状态
let sessionId = generateSessionId();
console.log('New session initialized:', sessionId);
let isProcessing = false;
let currentMessageDiv = null;
let interruptData = null;
let currentAbortController = null; // 用于中断当前请求
let currentReasoningDiv = null; // 当前深度思考区域
let isReasoningCollapsed = false; //思考是否折叠

// 跟踪消息 ID 和对应的 DOM 元素
// 用于处理 token 和 tool_calls ID 相同的情况
const messageIdMap = new Map();

// 暂存工具调用信息，等待 tool_output 合并显示
const pendingToolCalls = new Map();

// 跟踪每个消息的累积内容，用于 Markdown 渲染
const messageContentMap = new Map();

// 跟踪每个消息的累积思考内容，用于 Markdown 渲染
const reasoningContentMap = new Map();

// API 基础 URL
const API_BASE_URL = window.location.origin.includes('localhost') 
    ? 'http://localhost:7869' 
    : window.location.origin;

/**
 * 生成唯一的会话 ID
 */
function generateSessionId() {
    const id = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
    localStorage.setItem('chat_session_id', id);
    return id;
}

/**
 * 设置连接状态
 */
function setStatus(status, text) {
    const dot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    dot.className = 'status-dot';
    if (status === 'connecting') dot.classList.add('connecting');
    if (status === 'error') dot.classList.add('error');
    
    statusText.textContent = text;
}

/**
 * 设置输入框内容
 */
function setQuery(text) {
    const input = document.getElementById('messageInput');
    input.value = text;
    autoResize(input);
}

/**
 * 自动调整文本框高度
 */
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

/**
 * 处理键盘事件
 */
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

/**
 * 清空聊天记录
 */
function clearChat() {
    const container = document.getElementById('chatContainer');
    container.innerHTML = `
        <div class="welcome-message">
            <h2>👋 欢迎使用 AI Agent Chat</h2>
            <p>我可以帮您执行各种任务，包括计算、查询、数据处理等。</p>
            <div class="example-queries">
                <div class="example-query" onclick="setQuery('请你执行如下任务：\n1. 计算 10 + 10 的结果\n2. 将结果乘以 5\n3. 根据结果生成一个故事')">🧮 数学计算示例</div>
                <div class="example-query" onclick="setQuery('查询今天的天气情况')">🌤️ 天气查询</div>
                <div class="example-query" onclick="setQuery('帮我分析一下当前的市场趋势')">📊 数据分析</div>
            </div>
        </div>
    `;
    
    // 清空消息 ID 映射
    messageIdMap.clear();
    
    // 清空内容映射
    messageContentMap.clear();
    
    // 清空思考内容映射
    reasoningContentMap.clear();
    
    // 生成新会话 ID
    sessionId = generateSessionId();
    document.getElementById('sessionId').textContent = sessionId;
}

/**
 * 添加消息到聊天界面
 * @param {string} role - 角色 ('user' 或 'ai')
 * @param {string} content - 消息内容
 * @param {Object} options - 可选参数
 * @param {boolean} options.isToolCall - 是否是工具调用消息
 * @param {Object} options.toolData - 工具数据
 * @param {string} options.messageId - 消息 ID（用于跟踪）
 * @param {string} options.reasoningContent - 深度思考内容
 * @returns {HTMLElement} 消息 DOM 元素
 */
function addMessage(role, content, options = {}) {
    const container = document.getElementById('chatContainer');
    
    //移除欢迎消息
    const welcomeMsg = container.querySelector('.welcome-message');
    if (welcomeMsg) welcomeMsg.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    // 如果有 messageId，设置 data 属性以便后续查找
    if (options.messageId) {
        messageDiv.dataset.messageId = options.messageId;
    }
    
    const time = new Date().toLocaleTimeString('zh-CN');
    const avatarClass = role === 'user' ? 'user-avatar' : 'ai-avatar';
    const avatarText = role === 'user' ? '👤' : '🤖';
    const authorName = role === 'user' ? '用户' : 'AI 助手';
    
    let contentHtml = '';
    
    if (options.isToolCall && options.toolData) {
        //工具执行 -紧卡片式设计
        const toolName = options.toolData.toolCall ? options.toolData.toolCall.name : 'tool';
        const toolIcon = getToolIcon(toolName);
        
        // 获取输入参数的简短描述
        let argsSummary = '';
        if (options.toolData.toolCall && options.toolData.toolCall.args) {
            const args = options.toolData.toolCall.args;
            const argKeys = Object.keys(args);
            if (argKeys.length > 0) {
                const firstKey = argKeys[0];
                const firstValue = String(args[firstKey]).substring(0, 30);
                argsSummary = argKeys.length > 1 
                    ? `${firstKey}: ${firstValue}...` 
                    : `${firstKey}: ${firstValue}`;
            }
        }
        
        contentHtml = `
            <div class="tool-card" onclick="toggleToolCard(this)">
                <div class="tool-card-header">
                    <div class="tool-icon">${toolIcon}</div>
                    <div class="tool-info">
                        <div class="tool-name">${toolName}</div>
                        <div class="tool-args">${argsSummary}</div>
                    </div>
                    <div class="tool-status success">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                    </div>
                    <div class="tool-expand-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="6 9 12 15 18 9"></polyline>
                        </svg>
                    </div>
                </div>
                <div class="tool-card-details" style="display: none;">
        `;
        
        //显示工具调用信息
        if (options.toolData.toolCall) {
            const tool = options.toolData.toolCall;
            const argsStr = JSON.stringify(tool.args, null, 2);
            contentHtml += `
                <div class="tool-detail-section">
                    <div class="tool-detail-label">输入参数</div>
                    <pre class="tool-detail-code input">${escapeHtml(argsStr)}</pre>
                </div>
            `;
        }
        
        // 显示工具输出结果
        if (options.toolData.toolOutput) {
            const outputs = options.toolData.toolOutput;
            outputs.forEach(output => {
                const outputContent = typeof output.content === 'string' 
                    ? output.content 
                    : JSON.stringify(output.content, null, 2);
                contentHtml += `
                    <div class="tool-detail-section">
                        <div class="tool-detail-label">执行结果</div>
                        <pre class="tool-detail-code output">${escapeHtml(outputContent)}</pre>
                    </div>
                `;
            });
        }
        
        contentHtml += `
                </div>
            </div>
        `;
    } else {
        //普通消息
        contentHtml = `
            <div class="message-header">
                <div class="avatar ${avatarClass}">${avatarText}</div>
                <span class="message-author">${authorName}</span>
                <span class="message-time">${time}</span>
            </div>
        `;
        
        // 如果有深度思考内容，添加深度思考区域
        if (options.reasoningContent) {
            contentHtml += `
                <div class="reasoning-container expanded">
                    <div class="reasoning-header" onclick="toggleReasoning(this)">
                        <div class="reasoning-toggle">
                            <span class="reasoning-toggle-text">折叠思考</span>
                            <svg class="reasoning-toggle-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="transform: rotate(180deg);">
                                <polyline points="6 9 12 15 18 9"></polyline>
                            </svg>
                        </div>
                    </div>
                    <div class="reasoning-content-wrapper" style="display: block;">
                        <div class="reasoning-content">${options.reasoningContent}</div>
                    </div>
                </div>
            `;
        }
        
        contentHtml += `
            <div class="message-content">${content}</div>
        `;
    }
    
    messageDiv.innerHTML = contentHtml;
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
    
    return messageDiv;
}

/**
 * 切换工具卡片折叠状态
 * @param {HTMLElement} card - 点击的卡片元素
 */
function toggleToolCard(card) {
    const details = card.querySelector('.tool-card-details');
    const expandIcon = card.querySelector('.tool-expand-icon');
    
    if (details.style.display === 'none') {
        details.style.display = 'block';
        expandIcon.style.transform = 'rotate(180deg)';
        card.classList.add('expanded');
    } else {
        details.style.display = 'none';
        expandIcon.style.transform = 'rotate(0deg)';
        card.classList.remove('expanded');
    }
}

/**
 * 根据工具名称获取图标
 * @param {string} toolName - 工具名称
 * @returns {string} 图标字符
 */
function getToolIcon(toolName) {
    const iconMap = {
        'search': '🔍',
        'calculator': '🧮',
        'calc': '🧮',
        'math': '📐',
        'weather': '🌤️',
        'time': '⏰',
        'date': '📅',
        'file': '📄',
        'read': '📖',
        'write': '✍️',
        'edit': '✏️',
        'delete': '🗑️',
        'list': '📋',
        'get': '📥',
        'post': '📤',
        'put': '📤',
        'patch': '📤',
        'api': '🌐',
        'http': '🌐',
        'request': '📡',
        'fetch': '📡',
        'database': '🗄️',
        'db': '🗄️',
        'query': '🔎',
        'sql': '🗄️',
        'python': '🐍',
        'code': '💻',
        'exec': '⚡',
        'run': '▶️',
        'bash': '💻',
        'shell': '💻',
        'terminal': '💻',
        'git': '📦',
        'github': '🐙',
        'email': '📧',
        'mail': '📧',
        'send': '📤',
        'translate': '🌐',
        'convert': '🔄',
        'format': '📝',
        'parse': '🔍',
        'analyze': '📊',
        'chart': '📈',
        'graph': '📊',
        'plot': '📈',
        'image': '🖼️',
        'picture': '🖼️',
        'photo': '📷',
        'audio': '🔊',
        'video': '🎬',
        'music': '🎵',
        'map': '🗺️',
        'location': '📍',
        'navigate': '🧭',
        'browser': '🌐',
        'scrape': '🕷️',
        'crawl': '🕷️',
        'extract': '📤',
        'summarize': '📝',
        'summary': '📝'
    };
    
    // 尝试匹配工具名称
    const lowerName = toolName.toLowerCase();
    for (const [key, icon] of Object.entries(iconMap)) {
        if (lowerName.includes(key)) {
            return icon;
        }
    }
    
    return '🔧';
}

/**
 * HTML转义函数
 * @param {string} text - 需要转义的文本
 * @returns {string} 转义后的文本
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * 根据消息 ID 查找并移除消息
 * @param {string} messageId - 消息 ID
 * @returns {boolean} 是否成功移除
 */
function removeMessageById(messageId) {
    const container = document.getElementById('chatContainer');
    const messageDiv = container.querySelector(`[data-message-id="${messageId}"]`);
    
    if (messageDiv) {
        messageDiv.remove();
        return true;
    }
    return false;
}

/**
 * 显示打字指示器
 */
function showTypingIndicator() {
    const container = document.getElementById('chatContainer');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message ai typing';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="message-header">
            <div class="avatar ai-avatar">🤖</div>
            <span class="message-author">AI 助手</span>
            <span class="message-time">${new Date().toLocaleTimeString('zh-CN')}</span>
        </div>
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    container.appendChild(typingDiv);
    container.scrollTop = container.scrollHeight;
}

/**
 * 隐藏打字指示器
 */
function hideTypingIndicator() {
    const typingDiv = document.getElementById('typingIndicator');
    if (typingDiv) typingDiv.remove();
}

/**
 * 发送消息
 */
function sendMessage() {
    const input = document.getElementById('messageInput');
    const query = input.value.trim();
    
    if (!query || isProcessing) return;
    
    // 添加用户消息
    addMessage('user', query);
    input.value = '';
    input.style.height = 'auto';
    
    // 显示打字指示器
    showTypingIndicator();
    
    // 设置状态
    isProcessing = true;
    setStatus('connecting', '处理中...');
    document.getElementById('sendBtn').disabled = true;
    
    // 显示中断按钮，隐藏发送按钮
    document.getElementById('sendBtn').style.display = 'none';
    document.getElementById('abortBtn').style.display = 'flex';
    
    // 准备请求数据
    const internetSearch = document.getElementById('internetSearchCheckbox').checked;
    const deepThinking = document.getElementById('deepThinkingCheckbox').checked;
    const requestData = {
        query: query,
        session_id: sessionId,
        internet_search: internetSearch,
        deep_thinking: deepThinking
    };
    
    // 创建新的 AbortController
    currentAbortController = new AbortController();
    
    // 使用 Fetch API 发送 POST 请求并处理 SSE
    fetch(`${API_BASE_URL}/agent_chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        },
        body: JSON.stringify(requestData),
        signal: currentAbortController.signal
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        function readStream() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    finishProcessing();
                    return;
                }
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop() || '';
                
                lines.forEach(line => {
                    if (line.trim().startsWith('data:')) {
                        const data = line.trim().substring(5).trim();
                        try {
                            const event = JSON.parse(data);
                            handleStreamEvent(event);
                        } catch (e) {
                            console.error('解析事件数据失败:', e);
                        }
                    }
                });
                
                return readStream();
            });
        }
        
        return readStream();
    })
    .catch(error => {
        if (error.name === 'AbortError') {
            console.log('请求已被用户中断');
            hideTypingIndicator();
            // 不显示中断消息，保持界面简洁
        } else {
            console.error('请求失败:', error);
            hideTypingIndicator();
            addMessage('ai', `❌ 请求失败: ${error.message}`);
        }
        finishProcessing();
    });
}

/**
 * 中断当前请求
 */
function abortRequest() {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
}

/**
 *切换深度思考区域的折叠状态
 * @param {HTMLElement} header - 点击的深度思考头部元素
 */
function toggleReasoning(header) {
    console.log('切换深度思考折叠状态');
    const reasoningContainer = header.closest('.reasoning-container');
    if (!reasoningContainer) {
        console.log('没有找到深度思考区域');
        return;
    }
    
    const contentWrapper = reasoningContainer.querySelector('.reasoning-content-wrapper');
    const toggleText = reasoningContainer.querySelector('.reasoning-toggle-text');
    const toggleIcon = reasoningContainer.querySelector('.reasoning-toggle-icon');
    
    const isCollapsed = contentWrapper.style.display === 'none';
    
    if (isCollapsed) {
        // 展开状态
        console.log('切换到展开状态');
        contentWrapper.style.display = 'block';
        toggleText.textContent = '折叠思考';
        toggleIcon.style.transform = 'rotate(180deg)';
        reasoningContainer.classList.add('expanded');
    } else {
        // 折叠状态
        console.log('切换到折叠状态');
        contentWrapper.style.display = 'none';
        toggleText.textContent = '展开思考';
        toggleIcon.style.transform = 'rotate(0deg)';
        reasoningContainer.classList.remove('expanded');
    }
    
    //滚动到可见区域
    const container = document.getElementById('chatContainer');
    container.scrollTop = container.scrollHeight;
}

/**
 *处理流式事件
 * @param {Object} event - 事件对象
 */
function handleStreamEvent(event) {
    hideTypingIndicator();
    
    switch (event.event) {
        case 'token':
            //处理 token流
            console.log('处理 token 事件:', event.data);
            if (event.data && event.data.token) {
                const messageId = event.data.id;
                
                // 检查是否已经有相同 ID 的消息
                if (messageIdMap.has(messageId)) {
                    // 更新现有消息的内容
                    const existingDiv = messageIdMap.get(messageId);
                    const contentDiv = existingDiv.querySelector('.message-content');
                    
                    // 累积内容
                    let accumulatedContent = messageContentMap.get(messageId) || '';
                    accumulatedContent += event.data.token;
                    messageContentMap.set(messageId, accumulatedContent);
                    
                    if (contentDiv) {
                        // 使用 marked.js 渲染 Markdown
                        contentDiv.innerHTML = marked.parse(accumulatedContent);
                        console.log('更新消息内容:', event.data.token);
                    }
                } else {
                    // 创建新消息（没有深度思考内容的情况）
                    console.log('创建新消息');
                    currentMessageDiv = addMessage('ai', event.data.token, { messageId });
                    messageIdMap.set(messageId, currentMessageDiv);
                    // 初始化内容映射
                    messageContentMap.set(messageId, event.data.token);
                    
                    // 立即渲染 Markdown
                    const contentDiv = currentMessageDiv.querySelector('.message-content');
                    if (contentDiv) {
                        contentDiv.innerHTML = marked.parse(event.data.token);
                    }
                    console.log('新消息创建完成:', currentMessageDiv);
                }
                
                // 自动滚动
                const container = document.getElementById('chatContainer');
                container.scrollTop = container.scrollHeight;
            }
            break;
            
        case 'reasoning_token':
            //处理深度思考 token流
            console.log('处理 reasoning_token 事件:', event.data);
            if (event.data && event.data.token) {
                const messageId = event.data.id;
                
                // 检查是否已经有相同 ID 的消息
                if (messageIdMap.has(messageId)) {
                    // 更新现有消息的深度思考内容
                    const existingDiv = messageIdMap.get(messageId);
                    const reasoningContentDiv = existingDiv.querySelector('.reasoning-content');
                    const reasoningContentWrapper = existingDiv.querySelector('.reasoning-content-wrapper');
                    const reasoningContainer = existingDiv.querySelector('.reasoning-container');
                    const toggleText = existingDiv.querySelector('.reasoning-toggle-text');
                    const toggleIcon = existingDiv.querySelector('.reasoning-toggle-icon');
                    
                    // 累积思考内容
                    let accumulatedReasoning = reasoningContentMap.get(messageId) || '';
                    accumulatedReasoning += event.data.token;
                    reasoningContentMap.set(messageId, accumulatedReasoning);
                    
                    if (reasoningContentDiv) {
                        // 使用 marked.js 渲染 Markdown
                        reasoningContentDiv.innerHTML = marked.parse(accumulatedReasoning);
                        console.log('更新现有深度思考内容');
                    }
                    // 确保深度思考区域是展开状态
                    if (reasoningContentWrapper && reasoningContentWrapper.style.display === 'none') {
                        reasoningContentWrapper.style.display = 'block';
                        if (toggleText) toggleText.textContent = '折叠思考';
                        if (toggleIcon) toggleIcon.style.transform = 'rotate(180deg)';
                        if (reasoningContainer) reasoningContainer.classList.add('expanded');
                    }
                } else {
                    // 创建新消息并显示深度思考内容
                    console.log('创建新消息并显示深度思考内容');
                    currentMessageDiv = addMessage('ai', '', { 
                        messageId, 
                        reasoningContent: event.data.token 
                    });
                    messageIdMap.set(messageId, currentMessageDiv);
                    // 初始化思考内容映射
                    reasoningContentMap.set(messageId, event.data.token);
                    
                    // 立即渲染 Markdown
                    const reasoningContentDiv = currentMessageDiv.querySelector('.reasoning-content');
                    if (reasoningContentDiv) {
                        reasoningContentDiv.innerHTML = marked.parse(event.data.token);
                    }
                    console.log('新消息创建完成:', currentMessageDiv);
                }
                
                // 自动滚动
                const container = document.getElementById('chatContainer');
                container.scrollTop = container.scrollHeight;
            }
            break;
            
        case 'tool_calls':
            // 处理工具调用 - 暂存信息，等待 tool_output 合并显示
            if (event.data && event.data.tool_calls && event.data.id) {
                const messageId = event.data.id;
                
                // 检查是否有相同 ID 的 token 消息需要移除
                if (messageIdMap.has(messageId)) {
                    // 移除原始的 token 消息
                    removeMessageById(messageId);
                    messageIdMap.delete(messageId);
                }
                
                // 暂存工具调用信息
                pendingToolCalls.set(messageId, {
                    tool_calls: event.data.tool_calls,
                    messageId: messageId
                });
            }
            break;
            
        case 'tool_output':
            // 处理工具输出 - 合并 tool_calls 和 tool_output 显示
            if (event.data && event.data.tool_output) {
                const toolOutput = event.data.tool_output;
                
                // 查找匹配的工具调用
                let matchedToolCall = null;
                let matchedMessageId = null;
                
                // 根据 tool_call_id 匹配
                for (const [messageId, pendingData] of pendingToolCalls) {
                    const matchingCall = pendingData.tool_calls.find(
                        call => toolOutput.some(output => output.tool_call_id === call.id)
                    );
                    if (matchingCall) {
                        matchedToolCall = matchingCall;
                        matchedMessageId = messageId;
                        break;
                    }
                }
                
                // 如果找到匹配的工具调用，合并显示
                if (matchedToolCall && matchedMessageId) {
                    // 移除已处理的工具调用
                    pendingToolCalls.delete(matchedMessageId);
                    
                    // 添加合并后的工具消息
                    const toolDiv = addMessage('ai', '', {
                        isToolCall: true,
                        toolData: {
                            toolCall: matchedToolCall,
                            toolOutput: toolOutput
                        },
                        messageId: matchedMessageId
                    });
                    
                    // 记录消息
                    messageIdMap.set(matchedMessageId, toolDiv);
                } else {
                    // 如果没有找到匹配的工具调用，只显示输出
                    addMessage('ai', '', {
                        isToolCall: true,
                        toolData: {
                            toolOutput: toolOutput
                        }
                    });
                }
            }
            break;
            
        case '__interrupt__':
            // 处理中断
            interruptData = event.data.__interrupt__;
            showInterruptPanel(interruptData);
            break;
    }
}

/**
 * 显示中断面板
 * @param {Object} data - 中断数据
 */
function showInterruptPanel(data) {
    const panel = document.getElementById('interruptPanel');
    const content = document.getElementById('interruptContent');
    
    // 清空现有内容
    content.innerHTML = '';
    
    if (!data || !data.action_requests || data.action_requests.length === 0) {
        content.innerHTML = '<p>没有需要批准的操作</p>';
        panel.classList.add('active');
        finishProcessing();
        return;
    }
    
    // 遍历所有需要批准的操作
    data.action_requests.forEach((action, index) => {
        const actionDiv = document.createElement('div');
        actionDiv.className = 'interrupt-action';
        actionDiv.dataset.actionIndex = index;
        
        // 工具名称和图标
        const toolIcon = getToolIcon(action.name || 'tool');
        const headerHtml = `
            <div class="interrupt-action-header">
                <span class="interrupt-tool-icon">${toolIcon}</span>
                <span class="interrupt-tool-name">${escapeHtml(action.name || '未知工具')}</span>
            </div>
        `;
        
        // 工具描述
        const descHtml = action.description ? `
            <div class="interrupt-description">${escapeHtml(action.description)}</div>
        ` : '';
        
        // 工具参数（可编辑）
        const argsHtml = `
            <div class="interrupt-args-section">
                <div class="interrupt-section-label">工具参数:</div>
                <textarea 
                    class="interrupt-args-editor" 
                    id="argsEditor${index}"
                    rows="6"
                >${JSON.stringify(action.args || {}, null, 2)}</textarea>
                <div class="interrupt-args-hint">💡 您可以在上方直接编辑参数（JSON格式）</div>
            </div>
        `;
        
        // 拒绝原因输入框
        const rejectMessageHtml = `
            <div class="interrupt-reject-section" id="rejectSection${index}" style="display: none;">
                <div class="interrupt-section-label">拒绝原因（选填）:</div>
                <textarea 
                    class="interrupt-reject-message" 
                    id="rejectMessage${index}"
                    rows="3"
                    placeholder="请说明拒绝的原因或需要修改的地方...\n例如：不，这是错误的，因为...，应该这样做..."
                ></textarea>
            </div>
        `;
        
        actionDiv.innerHTML = headerHtml + descHtml + argsHtml + rejectMessageHtml;
        content.appendChild(actionDiv);
    });
    
    panel.classList.add('active');
    finishProcessing();
}

/**
 * 处理中断响应
 * @param {string} decision - 决策 ('approve', 'reject', 或 'edit')
 */
function handleInterrupt(decision) {
    const panel = document.getElementById('interruptPanel');
    const content = document.getElementById('interruptContent');
    const actions = content.querySelectorAll('.interrupt-action');
    
    if (actions.length === 0) {
        alert('没有找到需要处理的操作');
        return;
    }
    
    // 构建决策列表
    const decisions = [];
    let hasError = false;
    
    actions.forEach((actionDiv, index) => {
        const decisionObj = { type: decision };
        
        if (decision === 'edit') {
            // 获取编辑后的参数
            const argsEditor = document.getElementById(`argsEditor${index}`);
            try {
                const editedArgs = JSON.parse(argsEditor.value);
                const originalName = interruptData.action_requests[index].name;
                
                decisionObj.edited_action = {
                    name: originalName,
                    args: editedArgs
                };
            } catch (e) {
                alert(`参数 JSON 格式错误（操作 ${index + 1}）:\n${e.message}`);
                hasError = true;
                return;
            }
        } else if (decision === 'reject') {
            // 获取拒绝原因（可选）
            const rejectMessage = document.getElementById(`rejectMessage${index}`);
            if (rejectMessage && rejectMessage.value.trim()) {
                decisionObj.message = rejectMessage.value.trim();
            }
        }
        
        decisions.push(decisionObj);
    });
    
    if (hasError) return;
    
    // 关闭面板
    panel.classList.remove('active');
    
    // 发送恢复请求
    const requestData = {
        resume: {
            decisions: decisions
        },
        session_id: sessionId
    };
    
    // 添加用户操作提示消息
    let actionText = '';
    if (decision === 'approve') {
        actionText = '✅ 已批准操作继续执行';
    } else if (decision === 'reject') {
        actionText = '❌ 已拒绝操作';
        if (decisions[0].message) {
            actionText += `\n原因: ${decisions[0].message}`;
        }
    } else if (decision === 'edit') {
        actionText = '✏️ 已编辑参数并继续执行';
    }
    addMessage('user', actionText);
    
    isProcessing = true;
    setStatus('connecting', '处理中...');
    document.getElementById('sendBtn').disabled = true;
    
    // 显示中断按钮，隐藏发送按钮
    document.getElementById('sendBtn').style.display = 'none';
    document.getElementById('abortBtn').style.display = 'flex';
    
    showTypingIndicator();
    
    // 创建新的 AbortController
    currentAbortController = new AbortController();
    
    fetch(`${API_BASE_URL}/agent_chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        },
        body: JSON.stringify(requestData),
        signal: currentAbortController.signal
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        function readStream() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    finishProcessing();
                    return;
                }
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop() || '';
                
                lines.forEach(line => {
                    if (line.trim().startsWith('data:')) {
                        const data = line.trim().substring(5).trim();
                        try {
                            const event = JSON.parse(data);
                            handleStreamEvent(event);
                        } catch (e) {
                            console.error('解析事件数据失败:', e);
                        }
                    }
                });
                
                return readStream();
            });
        }
        
        return readStream();
    })
    .catch(error => {
        if (error.name === 'AbortError') {
            console.log('请求已被用户中断');
            hideTypingIndicator();
            // 不显示中断消息，保持界面简洁
        } else {
            console.error('请求失败:', error);
            hideTypingIndicator();
            addMessage('ai', `❌ 请求失败: ${error.message}`);
        }
        finishProcessing();
    });
}

/**
 * 切换拒绝原因输入框显示
 */
function toggleRejectReason() {
    const content = document.getElementById('interruptContent');
    const rejectConfirmArea = document.getElementById('rejectConfirmArea');
    const actions = content.querySelectorAll('.interrupt-action');
    
    // 切换所有操作的拒绝原因输入框
    let anyVisible = false;
    actions.forEach((actionDiv, index) => {
        const rejectSection = document.getElementById(`rejectSection${index}`);
        if (rejectSection) {
            const isHidden = rejectSection.style.display === 'none';
            rejectSection.style.display = isHidden ? 'block' : 'none';
            if (isHidden) anyVisible = true;
        }
    });
    
    // 显示或隐藏确认按钮区域
    if (rejectConfirmArea) {
        rejectConfirmArea.style.display = anyVisible ? 'block' : 'none';
    }
}

/**
 *完成处理
 */
function finishProcessing() {
    isProcessing = false;
    setStatus('ready', '就绪');
    document.getElementById('sendBtn').disabled = false;
    
    //恢复发送按钮，隐藏中断按钮
    document.getElementById('sendBtn').style.display = 'flex';
    document.getElementById('abortBtn').style.display = 'none';
    
    // 清理 AbortController
    currentAbortController = null;
}

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('sessionId').textContent = sessionId;
});
