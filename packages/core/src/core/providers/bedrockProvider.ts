/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
  type Message,
  type SystemContentBlock,
  type Tool,
  type ContentBlock,
} from '@aws-sdk/client-bedrock-runtime';
import {
  type GenerateContentParameters,
  type GenerateContentResponse,
  type CountTokensParameters,
  type CountTokensResponse,
  type EmbedContentParameters,
  type EmbedContentResponse,
  type Content,
  type Part,
  FinishReason,
} from '@google/genai';
import type { ContentGenerator } from '../contentGenerator.js';
import type { LlmRole } from '../../telemetry/llmRole.js';

export class BedrockContentGenerator implements ContentGenerator {
  private client: BedrockRuntimeClient;

  constructor(region?: string) {
    this.client = new BedrockRuntimeClient({
      region: region || process.env['AWS_REGION'] || 'us-east-1',
    });
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const messages = this.mapContentsToMessages(request.contents as Content[]);
    const system = this.mapSystemInstruction(request.config?.systemInstruction as any);
    const toolConfig = this.mapTools(request.config?.tools);

    const command = new ConverseCommand({
      modelId: request.model,
      messages,
      system,
      inferenceConfig: {
        maxTokens: request.config?.maxOutputTokens,
        temperature: request.config?.temperature,
        topP: request.config?.topP,
        stopSequences: request.config?.stopSequences,
      },
      toolConfig,
    });

    const response = await this.client.send(command);
    return this.mapResponse(response);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const messages = this.mapContentsToMessages(request.contents as Content[]);
    const system = this.mapSystemInstruction(request.config?.systemInstruction as any);
    const toolConfig = this.mapTools(request.config?.tools);

    const command = new ConverseStreamCommand({
      modelId: request.model,
      messages,
      system,
      inferenceConfig: {
        maxTokens: request.config?.maxOutputTokens,
        temperature: request.config?.temperature,
        topP: request.config?.topP,
        stopSequences: request.config?.stopSequences,
      },
      toolConfig,
    });

    const response = await this.client.send(command);
    return this.mapStreamResponse(response.stream);
  }

  async countTokens(_request: CountTokensParameters): Promise<CountTokensResponse> {
    return { totalTokens: 0 };
  }

  async embedContent(_request: EmbedContentParameters): Promise<EmbedContentResponse> {
    throw new Error('Embeddings not yet implemented for Bedrock provider.');
  }

  private mapContentsToMessages(contents: Content[]): Message[] {
    const messages: Message[] = [];

    for (const content of contents) {
      const role = content.role === 'model' ? 'assistant' : 'user';
      const contentBlocks: ContentBlock[] = [];

      for (const part of content.parts || []) {
        if (part.text) {
          contentBlocks.push({ text: part.text });
        }
        if (part.functionCall) {
          contentBlocks.push({
            toolUse: {
              toolUseId: `call_${Math.random().toString(36).substring(7)}`,
              name: part.functionCall.name,
              input: part.functionCall.args as any,
            },
          });
        }
        if (part.functionResponse) {
          contentBlocks.push({
            toolResult: {
              toolUseId: 'unknown',
              content: [{ json: part.functionResponse.response as any }],
              status: 'success',
            },
          });
        }
      }

      if (contentBlocks.length > 0) {
        messages.push({
          role,
          content: contentBlocks,
        });
      }
    }

    return messages;
  }

  private mapSystemInstruction(systemInstruction?: string | Part | Part[] | Content): SystemContentBlock[] | undefined {
    if (!systemInstruction) return undefined;

    let text = '';
    if (typeof systemInstruction === 'string') {
      text = systemInstruction;
    } else if (Array.isArray(systemInstruction)) {
      text = systemInstruction.map(p => p.text || '').join('\n');
    } else if ('parts' in systemInstruction && systemInstruction.parts) {
      text = systemInstruction.parts.map(p => p.text || '').join('\n');
    } else {
      text = (systemInstruction as Part).text || '';
    }

    return text ? [{ text }] : undefined;
  }

  private mapTools(tools?: any[]): { tools: Tool[] } | undefined {
    if (!tools || tools.length === 0) return undefined;

    const bedrockTools: Tool[] = [];
    for (const tool of tools) {
      if (tool.functionDeclarations) {
        for (const fd of tool.functionDeclarations) {
          bedrockTools.push({
            toolSpec: {
              name: fd.name,
              description: fd.description,
              inputSchema: {
                json: fd.parameters as any,
              },
            },
          });
        }
      }
    }

    return bedrockTools.length > 0 ? { tools: bedrockTools } : undefined;
  }

  private mapResponse(response: any): GenerateContentResponse {
    const parts: Part[] = [];
    if (response.output?.message?.content) {
      for (const block of response.output.message.content) {
        if (block.text) {
          parts.push({ text: block.text });
        }
        if (block.toolUse) {
          parts.push({
            functionCall: {
              name: block.toolUse.name,
              args: block.toolUse.input,
            },
          });
        }
      }
    }

    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          finishReason: this.mapFinishReason(response.stopReason),
        },
      ],
      usageMetadata: {
        promptTokenCount: response.usage?.inputTokens,
        candidatesTokenCount: response.usage?.outputTokens,
        totalTokenCount: response.usage?.totalTokens,
      },
    } as GenerateContentResponse;
  }

  private async *mapStreamResponse(stream: any): AsyncGenerator<GenerateContentResponse> {
    if (!stream) return;

    for await (const event of stream) {
      const parts: Part[] = [];
      let finishReason: any;
      let usage: any;

      if (event.contentBlockDelta?.delta?.text) {
        parts.push({ text: event.contentBlockDelta.delta.text });
      }

      if (event.messageStop?.stopReason) {
        finishReason = this.mapFinishReason(event.messageStop.stopReason);
      }

      if (event.metadata?.usage) {
        usage = event.metadata.usage;
      }

      if (parts.length > 0 || finishReason || usage) {
        yield {
          candidates: [
            {
              content: {
                role: 'model',
                parts,
              },
              finishReason,
            },
          ],
          usageMetadata: usage ? {
            promptTokenCount: usage.inputTokens,
            candidatesTokenCount: usage.outputTokens,
            totalTokenCount: usage.totalTokens,
          } : undefined,
        } as GenerateContentResponse;
      }
    }
  }

  private mapFinishReason(reason?: string): FinishReason {
    switch (reason) {
      case 'end_turn': return FinishReason.STOP;
      case 'max_tokens': return FinishReason.MAX_TOKENS;
      case 'stop_sequence': return FinishReason.STOP;
      case 'tool_use': return FinishReason.STOP;
      case 'content_filtered': return FinishReason.SAFETY;
      default: return FinishReason.OTHER;
    }
  }
}
