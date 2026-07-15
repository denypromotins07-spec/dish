/**
 * Stage 30: Chapter 4 - File 3
 * Audio Feedback System - Web Audio API
 * Zero RAM cost, pure oscillator synthesis
 */

type SoundType = 'orderFill' | 'orderPartial' | 'orderReject' | 'alert' | 'liquidationWarning';

interface AudioConfig {
  masterVolume: number;
  enabled: boolean;
}

class AudioFeedbackSystem {
  private audioContext: AudioContext | null = null;
  private config: AudioConfig = {
    masterVolume: 0.15, // Low volume to avoid annoyance
    enabled: true,
  };
  
  private gainNode: GainNode | null = null;
  private isInitialized = false;

  /**
   * Initialize audio context on user interaction (browser policy)
   */
  public init(): void {
    if (this.isInitialized) return;
    
    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.gainNode = this.audioContext.createGain();
      this.gainNode.connect(this.audioContext.destination);
      this.gainNode.gain.value = this.config.masterVolume;
      this.isInitialized = true;
    } catch (error) {
      console.warn('Web Audio API not supported or blocked:', error);
      this.config.enabled = false;
    }
  }

  /**
   * Play sound based on event type
   */
  public play(type: SoundType): void {
    if (!this.config.enabled || !this.isInitialized) return;
    
    // Ensure audio context is running (may be suspended by browser)
    if (this.audioContext?.state === 'suspended') {
      this.audioContext.resume();
    }

    switch (type) {
      case 'orderFill':
        this.playOrderFill();
        break;
      case 'orderPartial':
        this.playOrderPartial();
        break;
      case 'orderReject':
        this.playOrderReject();
        break;
      case 'alert':
        this.playAlert();
        break;
      case 'liquidationWarning':
        this.playLiquidationWarning();
        break;
    }
  }

  /**
   * Order Fill - Pleasant high-frequency click
   */
  private playOrderFill(): void {
    if (!this.audioContext || !this.gainNode) return;

    const now = this.audioContext.currentTime;
    
    // Main click oscillator
    const osc = this.audioContext.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(2800, now);
    osc.frequency.exponentialRampToValueAtTime(1200, now + 0.05);
    
    // Envelope for click
    const env = this.audioContext.createGain();
    env.gain.setValueAtTime(0, now);
    env.gain.linearRampToValueAtTime(0.4, now + 0.005);
    env.gain.exponentialRampToValueAtTime(0.01, now + 0.08);
    
    osc.connect(env);
    env.connect(this.gainNode);
    
    osc.start(now);
    osc.stop(now + 0.1);
  }

  /**
   * Partial Fill - Softer click
   */
  private playOrderPartial(): void {
    if (!this.audioContext || !this.gainNode) return;

    const now = this.audioContext.currentTime;
    
    const osc = this.audioContext.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(1800, now);
    osc.frequency.exponentialRampToValueAtTime(1000, now + 0.04);
    
    const env = this.audioContext.createGain();
    env.gain.setValueAtTime(0, now);
    env.gain.linearRampToValueAtTime(0.25, now + 0.005);
    env.gain.exponentialRampToValueAtTime(0.01, now + 0.06);
    
    osc.connect(env);
    env.connect(this.gainNode);
    
    osc.start(now);
    osc.stop(now + 0.08);
  }

  /**
   * Order Reject - Low buzz
   */
  private playOrderReject(): void {
    if (!this.audioContext || !this.gainNode) return;

    const now = this.audioContext.currentTime;
    
    const osc = this.audioContext.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(180, now);
    osc.frequency.linearRampToValueAtTime(120, now + 0.15);
    
    const env = this.audioContext.createGain();
    env.gain.setValueAtTime(0, now);
    env.gain.linearRampToValueAtTime(0.2, now + 0.01);
    env.gain.exponentialRampToValueAtTime(0.01, now + 0.15);
    
    // Lowpass filter to soften the sawtooth
    const filter = this.audioContext.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 800;
    
    osc.connect(filter);
    filter.connect(env);
    env.connect(this.gainNode);
    
    osc.start(now);
    osc.stop(now + 0.2);
  }

  /**
   * Alert - Two-tone chime
   */
  private playAlert(): void {
    if (!this.audioContext || !this.gainNode) return;

    const now = this.audioContext.currentTime;
    
    // First tone
    const osc1 = this.audioContext.createOscillator();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(1200, now);
    osc1.frequency.exponentialRampToValueAtTime(800, now + 0.15);
    
    const env1 = this.audioContext.createGain();
    env1.gain.setValueAtTime(0, now);
    env1.gain.linearRampToValueAtTime(0.3, now + 0.01);
    env1.gain.exponentialRampToValueAtTime(0.01, now + 0.2);
    
    osc1.connect(env1);
    env1.connect(this.gainNode);
    
    // Second tone (delayed)
    const osc2 = this.audioContext.createOscillator();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(1600, now + 0.12);
    osc2.frequency.exponentialRampToValueAtTime(1000, now + 0.27);
    
    const env2 = this.audioContext.createGain();
    env2.gain.setValueAtTime(0, now + 0.12);
    env2.gain.linearRampToValueAtTime(0.25, now + 0.13);
    env2.gain.exponentialRampToValueAtTime(0.01, now + 0.32);
    
    osc2.connect(env2);
    env2.connect(this.gainNode);
    
    osc1.start(now);
    osc1.stop(now + 0.22);
    osc2.start(now + 0.12);
    osc2.stop(now + 0.34);
  }

  /**
   * Liquidation Warning - Urgent pulsing tone
   */
  private playLiquidationWarning(): void {
    if (!this.audioContext || !this.gainNode) return;

    const now = this.audioContext.currentTime;
    
    for (let i = 0; i < 3; i++) {
      const t = now + i * 0.18;
      
      const osc = this.audioContext.createOscillator();
      osc.type = 'square';
      osc.frequency.setValueAtTime(600, t);
      osc.frequency.linearRampToValueAtTime(500, t + 0.12);
      
      const env = this.audioContext.createGain();
      env.gain.setValueAtTime(0, t);
      env.gain.linearRampToValueAtTime(0.15, t + 0.01);
      env.gain.exponentialRampToValueAtTime(0.01, t + 0.12);
      
      // Bandpass to make it less harsh
      const filter = this.audioContext.createBiquadFilter();
      filter.type = 'bandpass';
      filter.frequency.value = 550;
      filter.Q.value = 2;
      
      osc.connect(filter);
      filter.connect(env);
      env.connect(this.gainNode);
      
      osc.start(t);
      osc.stop(t + 0.15);
    }
  }

  /**
   * Update master volume
   */
  public setVolume(volume: number): void {
    this.config.masterVolume = Math.max(0, Math.min(1, volume));
    if (this.gainNode) {
      this.gainNode.gain.value = this.config.masterVolume;
    }
  }

  /**
   * Toggle audio on/off
   */
  public toggle(enabled: boolean): void {
    this.config.enabled = enabled;
    if (!enabled && this.audioContext?.state === 'running') {
      this.audioContext.suspend();
    } else if (enabled && this.audioContext?.state === 'suspended') {
      this.audioContext.resume();
    }
  }

  /**
   * Cleanup
   */
  public destroy(): void {
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
      this.isInitialized = false;
    }
  }
}

// Singleton instance
const audioFeedback = new AudioFeedbackSystem();

export default audioFeedback;
export type { SoundType, AudioConfig };
