setwd("/Users/hztan_1/Desktop/郭安平/")
library(dplyr)
library(ggplot2)
library(patchwork)
library(showtext)
showtext_auto()


full_data <- read.csv("v2_xgboost_feature_importance_v4.csv")

#自动筛选逻辑 (Automated Selection)
top_trend_row <- full_data %>%
  filter(feature == "rolling_mean_3") %>%
  arrange(desc(importance)) %>%
  slice(1) # 取第一名

# 销售第一的名字和厂家
target_trend_drug <- top_trend_row$drug_name
target_trend_mfr  <- top_trend_row$factory 

print(paste("趋势型代表:", target_trend_drug, "-", target_trend_mfr))

# 提取该药物 + 特定厂家的所有特征
df_trend <- full_data %>%
  filter(drug_name == target_trend_drug, factory == target_trend_mfr)

# 找出 Autoregression (Lag_1) 最强的 "药物+厂家" 组合
top_auto_row <- full_data %>%
  filter(feature == "lag_1") %>%
  arrange(desc(importance)) %>%
  slice(1)

# 获取第一的名字和厂家
target_auto_drug <- top_auto_row$drug_name
target_auto_mfr  <- top_auto_row$factory

print(paste("自回归型代表:", target_auto_drug, "-", target_auto_mfr))

# 提取数据
df_auto <- full_data %>%
  filter(drug_name == target_auto_drug, factory == target_auto_mfr)


# 排序因子 

df_trend$feature <- factor(df_trend$feature, levels = df_trend$feature[order(df_trend$importance)])
df_auto$feature  <- factor(df_auto$feature,  levels = df_auto$feature[order(df_auto$importance)])

# 3. 绘图 
library(ggplot2)
library(dplyr)
library(patchwork)

# 1. 极端字号优化函数 
create_barplot <- function(data, title_text, bar_color) {
  ggplot(data, aes(x = importance, y = feature)) +
    geom_col(fill = bar_color, width = 0.7) +
    geom_text(aes(label = round(importance, 2)), 
              hjust = -0.2, 
              size = 10,  
              fontface = "bold") + 
    scale_x_continuous(limits = c(0, max(data$importance)*1.4), expand = c(0, 0)) +
    labs(title = title_text, x = "Importance Score", y = "") +
    theme_classic() +
    theme(
      plot.title = element_text(face = "bold", size = 40, hjust = 0.5, margin = margin(b=20)),
      axis.text.y = element_text(size = 35, face = "bold", color = "black"),
      # X轴刻度设为 30
      axis.text.x = element_text(size = 30),
      # X轴标题设为 35
      axis.title.x = element_text(size = 35, face = "bold", margin = margin(t=20)),
      axis.line = element_line(size = 2), # 加粗坐标轴线
      axis.line.y = element_blank(),
      axis.ticks.y = element_blank(),
      plot.margin = margin(20, 40, 20, 20) # 增加边距
    )
}

# 2. 生成子图
p1 <- create_barplot(df_trend, "Trend-Driven Pattern", "#4A90E2")
p2 <- create_barplot(df_auto, "Autoregression-Driven Pattern", "#F5A623")

# 3. 拼图与保存 
final_plot <- p1 + p2 + 
  plot_annotation(
    tag_levels = 'A', 
    theme = theme(
      plot.tag = element_text(size = 50, face = "bold"), 
      plot.margin = margin(30, 30, 30, 30)
    )
  )

ggsave("Figure_Feature_Heterogeneity_FOR_WORD.tiff", 
       plot = final_plot, 
       width = 20,       
       height = 10, 
       units = "in", 
       dpi = 300,      
       compression = "lzw")

#0223
library(dplyr)
library(tidyr)
stratified_data <- read.csv("model_result_0223_haizhu.csv")[-c(1:3,18:19)]
# 1. 数据预清洗 (Data Pre-processing)
# 确保排除空行，并将所有指标列强制转换为数字型
clean_data <- stratified_data %>%
  filter(!is.na(mean_monthly_volume), mean_monthly_volume > 0) %>%
  mutate(across(starts_with("R2") | starts_with("SMAPE"), 
                ~ as.numeric(as.character(.))))

# 2. 分层与数据重组 (Stratification & Reshaping)
stratified_analysis <- clean_data %>%
  # 以销量中位数为界进行分层
  mutate(Group = ifelse(mean_monthly_volume >= median(mean_monthly_volume, na.rm=T), 
                        "High-Volume", "Low-Volume")) %>%
  # 选中所有核心指标列
  select(Group, 
         R2_XSP, SMAPE_XSP,
         R2_pro_final, SMAPE_pro_final, R2_SAR_final, SMAPE_SAR_final, R2_xgb_final, SMAPE_xgb_final,
         R2_pro_basic, SMAPE_pro_basic, R2_SAR_basic, SMAPE_SAR_basic, R2_xgb_basic, SMAPE_xgb_basic) %>%
  # 将宽表格转为长表格，便于按模型进行分组统计
  pivot_longer(
    cols = -Group,
    names_to = c(".value", "Model"),
    names_pattern = "(R2|SMAPE)_(.*)"
  )


# 3. 统计汇总 (Final Aggregation)
# 使用 Median R2 应对异常值，使用 Mean SMAPE 评估整体准确度
table_3_final <- stratified_analysis %>%
  group_by(Group, Model) %>%
  summarise(
    n = n(),
    Median_R2 = median(R2, na.rm = TRUE),
    Mean_SMAPE = mean(SMAPE, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  arrange(Group, desc(Model == "XSP"), desc(Median_R2)) %>%
  mutate(across(where(is.numeric), ~ round(., 3)))

########################
# 定义计算函数：基于预测误差降低估算库存降低
# 定义库存降低估算函数
calculate_ss_reduction <- function(baseline_smape, final_smape) {
  delta <- (baseline_smape - final_smape) / baseline_smape
  reduction <- 1 - sqrt(1 - delta)
  return(round(reduction * 100, 2))
}

# 对比对象 1: Basic Models 的平均表现 (代表传统方法)
avg_basic_smape_high_vol <- 65.36
final_xsp_smape_high_vol <- 40.8

# 对比对象 2: Optimized Single Models 的平均表现 (代表单模型优化极限)
# (41.1 + 49.1 + 47.5) / 3 = 45.9%
avg_optimized_smape_high_vol <- 45.9

# 计算库存节省潜力

# 1. 相比于传统方法 (Basic)，能省多少？
reduction_vs_basic <- calculate_ss_reduction(avg_basic_smape_high_vol, final_xsp_smape_high_vol)

# 2. 相比于优化后的单模型，还能多省多少？
reduction_vs_optimized <- calculate_ss_reduction(avg_optimized_smape_high_vol, final_xsp_smape_high_vol)

########
#0224
library(readxl)
library(dplyr)
library(extrafont)

data_vbp <- read_excel("model_result_andbasic_0224.xlsx", sheet = "3 final models") %>%
  select(4, 6)

data_vbp <- data_vbp %>%
  mutate(is_VBP = ifelse(is_VBP == "Y", "NND Drug", "Non-NND Drug"))

# 目的：证明 VBP 组和 Non-VBP 组的模型偏好不同
contingency_table <- table(data_vbp$is_VBP, data_vbp$selected_model)
chi_sq_result <- chisq.test(contingency_table)
#有没集采之间存在显著差异

# 3. 绘图数据准备 (计算百分比)
plot_data <- data_vbp %>%
  group_by(is_VBP, selected_model) %>%
  summarise(count = n(), .groups = 'drop') %>%
  group_by(is_VBP) %>%
  mutate(percentage = count / sum(count) * 100)


# 4. 画图 (堆叠条形图)
p <- ggplot(plot_data, aes(x = is_VBP, y = percentage, fill = selected_model)) +
  geom_bar(stat = "identity", position = "fill", width = 0.6) +
  
  # 百分比标签 (family = "Times New Roman")
  geom_text(aes(label = paste0(round(percentage, 1), "%")), 
            position = position_fill(vjust = 0.5), 
            size = 4, 
            color = "white", 
            fontface = "bold",
            family = "Times New Roman") +
  
  # 颜色和轴标签
  scale_y_continuous(labels = scales::percent) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = NULL,
       x = "Drug Category",
       y = "Proportion of Selected Models",
       fill = "Optimized_Single Model") +
  
  theme_minimal(base_family = "Times New Roman") + 
  theme(
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.position = "top",
    plot.margin = margin(10, 80, 10, 10) # 增加右边距
  ) +
  
  coord_cartesian(clip = "off") +
  annotate("text", x = 2.6, y = 0.5, label = p_val_text, 
           size = 4, 
         #  fontface = "italic", 
           hjust = 0,
           family = "Times New Roman")

##########
#生成图 4
library(ggplot2)
library(cowplot)
library(magick)

# 1. 读取 6 张图片文件

setwd("/Users/hztan_1/Desktop/郭安平/文章模型图片/")
file_paths <- c(
  "figure2_XGBoost_Prediction_麦考酚钠肠溶片_瑞士诺华.png", "figure3_XGBoost_Prediction_氟比洛芬凝胶贴膏_京泰德.png",  
  "figure4_Prophet_Prediction_复方黄柏液涂剂_鲁汉方.png", "figure5_Prophet_Prediction_肾衰宁片_山海关药业.png",  
  "figure6_SARIMAX_Prediction_腹膜透析液[乳酸盐]_华仁.png", "figure7_SARIMAX_Prediction_维生素B1片_信谊黄河.png"   
)

plot_list <- lapply(file_paths, function(path) {
  ggdraw() + draw_image(path)
})

# ======================================================
# 3. 拼图 (3行 x 2列)
# ======================================================
combined_plot <- plot_grid(
  plotlist = plot_list,
  ncol = 2, 
  nrow = 3,
  labels = c("A", "B", "C", "D", "E", "F"), 
  label_size = 8,      
  label_fontface = "bold",
  scale = 0.96         
)

# 5. 保存最终大图
ggsave("Figure_3_Combined_Cases.tiff", 
       plot = final_plot, 
       width = 16,       
       height = 18,     
       dpi = 300,      
       compression = "lzw")

