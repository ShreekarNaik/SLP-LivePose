export type CameraInfo = {
  cam_id: number;
  K: number[][];
  R: number[][];
  t: number[];
  width: number;
  height: number;
};

export type CameraCalib = {
  K: number[][];
  R: number[][];
  t: number[];
  image_size: [number, number]; // [width, height]
};

export type Skeleton = {
  joints: string[];
  edges: [number, number][];
};

export type MetaMessage = {
  type: "meta";
  cameras: Record<string, CameraCalib>;
  skeletons: {
    coco17: Skeleton;
    h36m17: Skeleton;
  };
  h36m_to_coco_subset: [number, number][]; // [h36m_idx, coco_idx]
};

export type SessionInfo = {
  name: string;
  fps: number;
  total_frames: number;
  cameras: CameraInfo[];
  has_ground_truth: boolean;
  joint_names: string[];
  skeleton: number[][];
};

export type SequencePreview = {
  subject: string;
  name: string;
  display_name: string;
  cam_ids: number[];
  has_ground_truth: boolean;
  fps: number;
  total_frames: number;
  seq_path: string;
};

export type ScanResponse = {
  kind: string;
  root: string;
  sequences: SequencePreview[];
};

export type Pose2D = {
  kp: number[][]; // [J, 2]
  conf: number[]; // [J]
  detected: boolean;
};

export type FrameMessage = {
  type: "frame";
  sequence: number;
  fps: number;
  frame_index: number;
  total_frames: number;
  timestamp: number;
  cam_ids: number[];
  blobs: { cam_id: number; offset: number; length: number }[];
  poses_2d: Record<string, Pose2D>;
  points_3d: (number | null)[][]; // [J, 3], null where invalid
  valid_3d: boolean[];
  gt_3d: (number | null)[][] | null;
  stage_timings_ms: Record<string, number>;
};
