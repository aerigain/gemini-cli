import { describe, it, expect } from 'vitest';
import { BedrockContentGenerator } from './bedrockProvider.js';
import { LlmRole } from '../../telemetry/llmRole.js';

describe('Bedrock Integration Validation', () => {
  it('should successfully communicate with Bedrock Nova 2 Lite', async () => {
    const generator = new BedrockContentGenerator('us-east-1');
    const request = {
      model: 'us.amazon.nova-lite-v1:0',
      contents: [{ role: 'user', parts: [{ text: 'Respond with exactly: VALIDATED' }] }],
    };

    console.log('Attempting real Bedrock call...');
    try {
      const response = await generator.generateContent(request as any, 'val-id', LlmRole.MAIN);
      console.log('Bedrock Response:', JSON.stringify(response, null, 2));
      expect(response.candidates?.[0].content?.parts?.[0].text).toContain('VALIDATED');
    } catch (error) {
      console.error('Bedrock Call Failed:', error);
      throw error;
    }
  }, 30000); // 30s timeout
});
