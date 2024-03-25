//
//  HeadMetas.swift
//  openpifpafios
//
//  Created by Alexander Ivkin on 3/25/24.
//

import Foundation

let defaultMetas = HeadMetas(
    cif: CifMeta(
        keypoints: ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel", "f_24", "f_25", "f_26", "f_27", "f_28", "f_29", "f_30", "f_31", "f_32", "f_33", "f_34", "f_35", "f_36", "f_37", "f_38", "f_39", "f_40", "f_41", "f_42", "f_43", "f_44", "f_45", "f_46", "f_47", "f_48", "f_49", "f_50", "f_51", "f_52", "f_53", "f_54", "f_55", "f_56", "f_57", "f_58", "f_59", "f_60", "f_61", "f_62", "f_63", "f_64", "f_65", "f_66", "f_67", "f_68", "f_69", "f_70", "f_71", "f_72", "f_73", "f_74", "f_75", "f_76", "f_77", "f_78", "f_79", "f_80", "f_81", "f_82", "f_83", "f_84", "f_85", "f_86", "f_87", "f_88", "f_89", "f_90", "f_91", "lh_92", "lh_93", "lh_94", "lh_95", "lh_96", "lh_97", "lh_98", "lh_99", "lh_100", "lh_101", "lh_102", "lh_103", "lh_104", "lh_105", "lh_106", "lh_107", "lh_108", "lh_109", "lh_110", "lh_111", "lh_112", "rh_113", "rh_114", "rh_115", "rh_116", "rh_117", "rh_118", "rh_119", "rh_120", "rh_121", "rh_122", "rh_123", "rh_124", "rh_125", "rh_126", "rh_127", "rh_128", "rh_129", "rh_130", "rh_131", "rh_132", "rh_133"],
        stride: 16,
        score_weights: []
    )
)

struct HeadMetas {
    let cif: CifMeta
}

struct CifMeta {
    let keypoints: [String]
    let stride: Int
    let score_weights: [Double]
}

struct CafMeta {
    let stride: Int
    let skeleton: [[Int]]
}
