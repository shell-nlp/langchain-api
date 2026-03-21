"use client";
import "@copilotkit/react-core/v2/styles.css";
import "./ui/style.css";
import {
  useRenderTool,
  useDefaultRenderTool,
  useConfigureSuggestions,
  CopilotChat,
} from "@copilotkit/react-core/v2";
import { z } from "zod";
import { CopilotKit } from "@copilotkit/react-core";
import conditionsToChinese from "./utils/weather";

const AgenticChat = () => {
  return (
    <CopilotKit
      runtimeUrl="/api/copilotkit"
      showDevConsole={false}
      agent="sample_agent"
      // headers 用来添加自定义的请求头，比如 Authorization
      headers={{
        Authorization:
          "Bearer " + (localStorage.getItem("token") || "test_token"),
      }}
      // properties 用来传入其它的参数，比如是否开启互联网搜索、是否开启深度思考等
      properties={{
        internet_search: false,
        deep_thinking: true,
      }}
    >
      <Chat />
    </CopilotKit>
  );
};

const Chat = () => {
  // 为其他所有工具提供通用UI
  useDefaultRenderTool({
    render: ({ name, args, status, result }) => {
      const isComplete = status === "complete";
      return (
        <div
          className={`rounded-xl mt-6 mb-4 max-w-md w-full border border-gray-200 shadow-sm ${
            isComplete
              ? "bg-white"
              : "bg-gradient-to-br from-indigo-500 to-purple-600"
          }`}
        >
          <div className={`p-4 ${!isComplete && "text-white"}`}>
            <div className="flex items-center gap-3">
              {!isComplete && <span className="animate-spin text-lg">⚙️</span>}
              {isComplete && (
                <span className="w-6 h-6 rounded-full bg-green-500 flex items-center justify-center text-white text-sm">
                  ✓
                </span>
              )}
              <h4
                className={`font-semibold ${isComplete ? "text-gray-800" : "text-white"}`}
              >
                {name}
              </h4>
            </div>
            {!isComplete && (
              <p className="text-white/70 text-sm mt-2">正在执行中...</p>
            )}
            {isComplete && (
              <>
                {args && Object.keys(args).length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-100">
                    <p className="text-xs text-gray-500 mb-2 font-medium">
                      输入参数
                    </p>
                    <pre className="text-xs text-gray-600 bg-gray-50 rounded-lg p-3 overflow-x-auto">
                      {JSON.stringify(args, null, 2)}
                    </pre>
                  </div>
                )}
                {result && (
                  <div className="mt-3 pt-3 border-t border-gray-100">
                    <p className="text-xs text-gray-500 mb-2 font-medium">
                      输出结果
                    </p>
                    {typeof result === "string" ? (
                      <pre className="text-xs text-gray-600 bg-gray-50 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap">
                        {result}
                      </pre>
                    ) : (
                      <pre className="text-xs text-gray-600 bg-gray-50 rounded-lg p-3 overflow-x-auto">
                        {JSON.stringify(result, null, 2)}
                      </pre>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      );
    },
  });
  // 为特定工具提供自定义UI渲染
  useRenderTool({
    name: "get_weather",
    parameters: z.object({
      location: z.string(),
    }),
    render: ({ args, result, status }: any) => {
      // result: {"temperature": 20, "conditions": "sunny", "humidity": 50, "wind_speed": 10, "feelsLike": 25}
      // result 是字符串，需要解析为 JSON 对象
      if (typeof result === "string") {
        result = JSON.parse(result);
      }
      if (status !== "complete") {
        return (
          <div className=" bg-[#667eea] text-white p-4 rounded-lg max-w-md">
            <span className="animate-spin">⚙️ Retrieving weather...</span>
          </div>
        );
      }

      const weatherResult: WeatherToolResult = {
        temperature: result?.temperature || 0,
        conditions: result?.conditions || "clear",
        humidity: result?.humidity || 0,
        windSpeed: result?.wind_speed || 0,
        feelsLike: result?.feels_like || result?.temperature || 0,
      };

      const themeColor = getThemeColor(weatherResult.conditions);
      // 将 conditions 转换为中文描述
      weatherResult.conditions = conditionsToChinese(weatherResult.conditions);
      console.log("weatherResult.conditions", weatherResult.conditions);
      return (
        <WeatherCard
          location={args.location}
          themeColor={themeColor}
          result={weatherResult}
          status={status || "complete"}
        />
      );
    },
  });

  useConfigureSuggestions({
    suggestions: [
      {
        title: "天气情况",
        message: "今天天气怎么样？",
      },
      {
        title: "工具详情",
        message: "你有那些工具可用？",
      },

      {
        title: "时间信息",
        message: "你今天是几号？星期几？",
      },
    ],
    available: "always",
  });

  return (
    <div className="flex justify-center min-h-screen w-full p-4">
      <div className="w-full md:w-4/5 h-screen rounded-lg">
        <CopilotChat
          agentId="sample_agent"
          className="h-full rounded-2xl max-w-6xl mx-auto"
          labels={{
            welcomeMessageText: "今天我能为您做些什么?",
            chatInputPlaceholder: "请输入您的问题...",
            chatDisclaimerText: "AI可能会出错，请核实重要信息。",
          }}
        />
      </div>
    </div>
  );
};

interface WeatherToolResult {
  temperature: number;
  conditions: string;
  humidity: number;
  windSpeed: number;
  feelsLike: number;
}

function getThemeColor(conditions: string): string {
  const conditionLower = conditions.toLowerCase();
  if (conditionLower.includes("clear") || conditionLower.includes("sunny")) {
    return "#667eea";
  }
  if (conditionLower.includes("rain") || conditionLower.includes("storm")) {
    return "#4A5568";
  }
  if (conditionLower.includes("cloud")) {
    return "#718096";
  }
  if (conditionLower.includes("snow")) {
    return "#63B3ED";
  }
  return "#764ba2";
}

function WeatherCard({
  location,
  themeColor,
  result,
  status,
}: {
  location?: string;
  themeColor: string;
  result: WeatherToolResult;
  status: "inProgress" | "executing" | "complete";
}) {
  return (
    <div
      data-testid="weather-card"
      style={{ backgroundColor: themeColor }}
      className="rounded-xl mt-6 mb-4 max-w-md w-full"
    >
      <div className="bg-white/20 p-4 w-full">
        <div className="flex items-center justify-between">
          <div>
            <h3
              data-testid="weather-city"
              className="text-xl font-bold text-white capitalize"
            >
              {location}
            </h3>
            <p className="text-white">当前天气</p>
          </div>
          <WeatherIcon conditions={result.conditions} />
        </div>

        <div className="mt-4 flex items-end justify-between">
          <div className="text-3xl font-bold text-white">
            <span className="">{result.temperature}° C</span>
            <span className="text-sm text-white/50">
              {" / "}
              {((result.temperature * 9) / 5 + 32).toFixed(1)}° F
            </span>
          </div>
          <div className="text-sm text-white capitalize">
            {result.conditions}
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-white">
          <div className="grid grid-cols-3 gap-2 text-center">
            <div data-testid="weather-humidity">
              <p className="text-white text-xs">湿度</p>
              <p className="text-white font-medium">{result.humidity}%</p>
            </div>
            <div data-testid="weather-wind">
              <p className="text-white text-xs">风速</p>
              <p className="text-white font-medium">{result.windSpeed} mph</p>
            </div>
            <div data-testid="weather-feels-like">
              <p className="text-white text-xs">体感温度</p>
              <p className="text-white font-medium">{result.feelsLike}°</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function WeatherIcon({ conditions }: { conditions: string }) {
  if (!conditions) return null;

  if (
    conditions.toLowerCase().includes("clear") ||
    conditions.toLowerCase().includes("sunny")
  ) {
    return <SunIcon />;
  }

  if (
    conditions.toLowerCase().includes("rain") ||
    conditions.toLowerCase().includes("drizzle") ||
    conditions.toLowerCase().includes("snow") ||
    conditions.toLowerCase().includes("thunderstorm")
  ) {
    return <RainIcon />;
  }

  if (
    conditions.toLowerCase().includes("fog") ||
    conditions.toLowerCase().includes("cloud") ||
    conditions.toLowerCase().includes("overcast")
  ) {
    return <CloudIcon />;
  }

  return <CloudIcon />;
}

// Simple sun icon for the weather card
function SunIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="currentColor"
      className="w-14 h-14 text-yellow-200"
    >
      <circle cx="12" cy="12" r="5" />
      <path
        d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"
        strokeWidth="2"
        stroke="currentColor"
      />
    </svg>
  );
}

function RainIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="currentColor"
      className="w-14 h-14 text-blue-200"
    >
      {/* Cloud */}
      <path
        d="M7 15a4 4 0 0 1 0-8 5 5 0 0 1 10 0 4 4 0 0 1 0 8H7z"
        fill="currentColor"
        opacity="0.8"
      />
      {/* Rain drops */}
      <path
        d="M8 18l2 4M12 18l2 4M16 18l2 4"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  );
}

function CloudIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="currentColor"
      className="w-14 h-14 text-gray-200"
    >
      <path
        d="M7 15a4 4 0 0 1 0-8 5 5 0 0 1 10 0 4 4 0 0 1 0 8H7z"
        fill="currentColor"
      />
    </svg>
  );
}

export default AgenticChat;
